import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class TaskPriorityQueue:
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self._queue: List[Dict[str, Any]] = []

    def push(self, task_id: str, priority_score: float, task_type: str, payload: Dict[str, Any]) -> bool:
        if len(self._queue) >= self.capacity:
            logging.warning("Task priority queue reached maximum capacity.")
            return False
        
        task_entry = {
            "task_id": task_id,
            "priority_score": float(priority_score),
            "task_type": task_type,
            "payload": payload,
            "timestamp": time.time(),
            "status": "QUEUED"
        }
        self._queue.append(task_entry)
        self._sort_queue()
        return True

    def pop(self) -> Optional[Dict[str, Any]]:
        if not self._queue:
            return None
        return self._queue.pop(0)

    def _sort_queue(self) -> None:
        self._queue.sort(key=lambda item: item["priority_score"], reverse=True)

    def peek(self) -> Optional[Dict[str, Any]]:
        return self._queue[0] if self._queue else None

    def size(self) -> int:
        return len(self._queue)


class WorkingMemoryBuffer:
    def __init__(self, buffer_size: int = 256):
        self.buffer_size = buffer_size
        self.memory_stream = deque(maxlen=buffer_size)

    def write_trace(self, trace_id: str, vector_state: np.ndarray, metadata: Dict[str, Any]) -> None:
        trace_entry = {
            "trace_id": trace_id,
            "vector_state": vector_state,
            "metadata": metadata,
            "timestamp": time.time()
        }
        self.memory_stream.append(trace_entry)

    def get_recent_traces(self, k: int = 5) -> List[Dict[str, Any]]:
        return list(self.memory_stream)[-k:]


class TaskManager:
    def __init__(
        self, 
        state_dim: int = 128, 
        max_active_tasks: int = 10, 
        scheduler_rate_hz: float = 100.0
    ):
        self.state_dim = state_dim
        self.max_active_tasks = max_active_tasks
        self.scheduler_rate_hz = scheduler_rate_hz
        
        self.queue = TaskPriorityQueue(capacity=500)
        self.working_memory = WorkingMemoryBuffer(buffer_size=128)
        
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.completed_tasks: List[Dict[str, Any]] = []
        
        self.weights_task_valuation = np.random.randn(state_dim, 1) * 0.01
        self.cognitive_focus_state = np.zeros((1, state_dim))
        
        logging.info("Cognitive TaskManager successfully initialized.")

    def compute_dynamic_priority(
        self, 
        base_priority: float, 
        context_vector: np.ndarray, 
        urgency_factor: float = 1.0
    ) -> float:
        if context_vector.shape[-1] != self.state_dim:
            raise ValueError(f"Context vector dimension mismatch. Expected {self.state_dim}, got {context_vector.shape[-1]}")
            
        cognitive_alignment = float(np.dot(context_vector, self.weights_task_valuation))
        dynamic_score = (base_priority * 0.4) + (cognitive_alignment * 0.3) + (urgency_factor * 0.3)
        return float(np.clip(dynamic_score, 0.0, 100.0))

    def register_task(
        self, 
        task_id: str, 
        task_type: str, 
        base_priority: float, 
        context_vector: np.ndarray, 
        payload: Dict[str, Any],
        urgency: float = 1.0
    ) -> bool:
        priority_score = self.compute_dynamic_priority(base_priority, context_vector, urgency)
        success = self.queue.push(
            task_id=task_id, 
            priority_score=priority_score, 
            task_type=task_type, 
            payload=payload
        )
        if success:
            logging.info(f"Task '{task_id}' [{task_type}] registered with computed priority: {priority_score:.4f}")
        return success

    def update_cognitive_focus(self, new_state: np.ndarray) -> None:
        if new_state.shape[-1] != self.state_dim:
            raise ValueError(f"State dimension mismatch. Expected {self.state_dim}, got {new_state.shape[-1]}")
        self.cognitive_focus_state = new_state
        self.working_memory.write_trace(
            trace_id=f"focus_step_{time.time()}", 
            vector_state=new_state, 
            metadata={"status": "FOCUS_UPDATED"}
        )

    def dispatch_next_task(self) -> Optional[Dict[str, Any]]:
        if len(self.active_tasks) >= self.max_active_tasks:
            logging.warning("Cannot dispatch task: Maximum active execution capacity reached.")
            return None

        next_task = self.queue.pop()
        if not next_task:
            return None

        task_id = next_task["task_id"]
        next_task["status"] = "EXECUTING"
        next_task["start_time"] = time.time()
        
        self.active_tasks[task_id] = next_task
        logging.info(f"Dispatched task '{task_id}' for cognitive processing.")
        return next_task

    def complete_task(self, task_id: str, execution_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if task_id not in self.active_tasks:
            logging.error(f"Task completion failure: Task '{task_id}' not found in active execution table.")
            return None

        task_data = self.active_tasks.pop(task_id)
        task_data["status"] = "COMPLETED"
        task_data["end_time"] = time.time()
        task_data["execution_duration"] = task_data["end_time"] - task_data["start_time"]
        task_data["result"] = execution_result

        self.completed_tasks.append(task_data)
        
        self.working_memory.write_trace(
            trace_id=f"complete_{task_id}",
            vector_state=self.cognitive_focus_state,
            metadata={"task_id": task_id, "duration": task_data["execution_duration"]}
        )
        logging.info(f"Task '{task_id}' successfully completed in {task_data['execution_duration']:.4f}s.")
        return task_data

    def evaluate_system_workload(self) -> Dict[str, Any]:
        return {
            "queued_tasks_count": self.queue.size(),
            "active_tasks_count": len(self.active_tasks),
            "completed_tasks_count": len(self.completed_tasks),
            "working_memory_traces": len(self.working_memory.memory_stream),
            "highest_pending_priority": self.queue.peek()["priority_score"] if self.queue.peek() else 0.0
        }

    def scheduler_step(self, sensory_context: np.ndarray) -> Dict[str, Any]:
        self.update_cognitive_focus(sensory_context)
        
        dispatched_item = None
        if self.queue.size() > 0 and len(self.active_tasks) < self.max_active_tasks:
            dispatched_item = self.dispatch_next_task()

        workload_metrics = self.evaluate_system_workload()
        return {
            "dispatched_task": dispatched_item,
            "system_telemetry": workload_metrics,
            "current_focus_norm": float(np.linalg.norm(self.cognitive_focus_state))
        }


if __name__ == "__main__":
    task_manager = TaskManager(state_dim=128, max_active_tasks=3)

    dummy_context_vector = np.random.randn(1, 128)

    task_manager.register_task(
        task_id="TASK_NAV_001",
        task_type="NAVIGATION",
        base_priority=75.0,
        context_vector=dummy_context_vector,
        payload={"target_coordinates": [12.5, 43.2, 0.0]},
        urgency=1.5
    )

    task_manager.register_task(
        task_id="TASK_COMM_002",
        task_type="COMMUNICATION",
        base_priority=50.0,
        context_vector=dummy_context_vector,
        payload={"packet_size": 1024, "protocol": "5G_DIRECT"},
        urgency=0.8
    )

    scheduler_output = task_manager.scheduler_step(sensory_context=dummy_context_vector)
    
    print("\n--- Task Manager Scheduler Output ---")
    print(f"Dispatched Task ID: {scheduler_output['dispatched_task']['task_id'] if scheduler_output['dispatched_task'] else 'None'}")
    print(f"Active Tasks: {scheduler_output['system_telemetry']['active_tasks_count']}")
    print(f"Queued Tasks: {scheduler_output['system_telemetry']['queued_tasks_count']}")

    if scheduler_output["dispatched_task"]:
        dispatched_id = scheduler_output["dispatched_task"]["task_id"]
        task_manager.complete_task(dispatched_id, execution_result={"status_code": 200, "message": "Execution nominal"})

    final_telemetry = task_manager.evaluate_system_workload()
    print("\n--- Final System Telemetry ---")
    print(f"Completed Tasks: {final_telemetry['completed_tasks_count']}")
