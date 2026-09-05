"""
Tests for the production celery-beat schedule (crontab-based, configurable
via Settings) and the new scheduled_retrain_all_models task.
"""
from celery.schedules import crontab


def test_beat_schedule_uses_real_crontab_not_raw_interval():
    from app.workers.celery_app import celery_app

    schedule = celery_app.conf.beat_schedule
    assert "daily-drift-check" in schedule
    assert "weekly-model-retrain" in schedule

    for entry in schedule.values():
        # A raw int/float schedule (old placeholder) means "N seconds after
        # beat started" -- not a fixed time of day. Every entry here must be
        # a real crontab, which fixes an actual wall-clock time.
        assert isinstance(entry["schedule"], crontab), f"{entry['task']} is not using a crontab schedule"


def test_drift_check_schedule_matches_configured_time():
    from app.core.config import get_settings
    from app.workers.celery_app import celery_app

    settings = get_settings()
    entry = celery_app.conf.beat_schedule["daily-drift-check"]
    sched = entry["schedule"]
    assert settings.DRIFT_CHECK_HOUR_UTC in sched.hour
    assert settings.DRIFT_CHECK_MINUTE_UTC in sched.minute


def test_retrain_schedule_matches_configured_time_and_day():
    from app.core.config import get_settings
    from app.workers.celery_app import celery_app

    settings = get_settings()
    entry = celery_app.conf.beat_schedule["weekly-model-retrain"]
    sched = entry["schedule"]
    assert settings.MODEL_RETRAIN_HOUR_UTC in sched.hour
    assert settings.MODEL_RETRAIN_MINUTE_UTC in sched.minute


def test_drift_check_task_is_the_registered_task_name():
    from app.workers.celery_app import celery_app

    entry = celery_app.conf.beat_schedule["daily-drift-check"]
    assert entry["task"] == "app.workers.tasks.calculate_drift_metrics"
    assert entry["task"] in celery_app.tasks


def test_retrain_task_is_the_registered_task_name():
    from app.workers.celery_app import celery_app

    entry = celery_app.conf.beat_schedule["weekly-model-retrain"]
    assert entry["task"] == "app.workers.tasks.scheduled_retrain_all_models"
    assert entry["task"] in celery_app.tasks


def test_scheduled_drift_check_creates_its_own_trackable_job(client, admin_token):
    """
    A celery-beat-triggered drift check has no pre-existing job_id from an
    API call -- it must create and track its OWN BackgroundJob row (via
    _create_and_run_job) so it's still visible through GET /jobs/{id},
    giving real observability into scheduled runs.
    """
    from app.workers.tasks import calculate_drift_metrics

    result = calculate_drift_metrics.apply(args=(None, 500)).get()
    assert isinstance(result, dict)
    assert "n_metrics_computed" in result

    # Find the job this run created and confirm it's marked SUCCESS.
    from app.models.domain import BackgroundJob

    override_gen_fn = next(iter(client.app.dependency_overrides.values()))
    db = next(override_gen_fn())
    jobs = db.query(BackgroundJob).filter(BackgroundJob.task_name == "calculate_drift_metrics").all()
    assert len(jobs) == 1
    assert jobs[0].status == "SUCCESS"
    assert jobs[0].params.get("trigger") == "beat"


def test_scheduled_retrain_all_models_tracks_each_model_type_independently(client, admin_token):
    """No synthetic data in this fresh test DB -- each of the 4 model
    trainings will report 'not enough data' internally (a valid outcome,
    not a crash), and each must still get its own tracked BackgroundJob row
    so a failure in one doesn't obscure the others."""
    from app.workers.tasks import scheduled_retrain_all_models

    results = scheduled_retrain_all_models.apply(args=()).get()
    assert set(results.keys()) == {"denial_risk", "denial_reason", "appeal_success", "anomaly_detection"}

    from app.models.domain import BackgroundJob

    override_gen_fn = next(iter(client.app.dependency_overrides.values()))
    db = next(override_gen_fn())
    jobs = db.query(BackgroundJob).filter(BackgroundJob.task_name == "scheduled_retrain_all_models").all()
    assert len(jobs) == 4
    assert all(j.status == "SUCCESS" for j in jobs)  # "not enough data" is a successful task completion
    model_types_seen = {j.params["model_type"] for j in jobs}
    assert model_types_seen == {"denial_risk", "denial_reason", "appeal_success", "anomaly_detection"}
