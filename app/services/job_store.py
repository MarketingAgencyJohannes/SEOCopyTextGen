import json
import uuid
from datetime import datetime, timezone

from app.database import Job, get_session


def create_job(agent: str) -> str:
    job_id = str(uuid.uuid4())
    with get_session() as session:
        job = Job(
            id=job_id,
            agent=agent,
            status="pending",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        session.add(job)
        session.commit()
    return job_id


def update_job(
    job_id: str,
    status: str,
    result: dict | None = None,
    error: str | None = None,
) -> None:
    with get_session() as session:
        job = session.get(Job, job_id)
        if job is None:
            return
        job.status = status
        job.updated_at = datetime.now(timezone.utc)
        if result is not None:
            job.result_json = json.dumps(result)
        if error is not None:
            job.error_message = error
        session.commit()


def get_job(job_id: str) -> dict | None:
    with get_session() as session:
        job = session.get(Job, job_id)
        if job is None:
            return None
        return job.to_dict()
