"""Tests for WeatherRepository.

This module tests the weather repository with an in-memory SQLite database
to ensure proper CRUD operations and query functionality.
"""

from datetime import datetime, timedelta

import pytest
from sqlmodel import Session, SQLModel, create_engine

from app.db.models import WeatherApiCall
from app.db.repositories.weather_repository import WeatherRepository


@pytest.fixture(scope="function")
def test_engine():
    """Create test database engine."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    SQLModel.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture(scope="function")
def test_session(test_engine):
    """Create test database session."""
    with Session(test_engine) as session:
        yield session
        session.rollback()


@pytest.fixture(scope="function")
def repository(test_session):
    """Create repository instance."""
    return WeatherRepository(test_session)


@pytest.fixture
def sample_weather_data():
    """Sample weather API response data."""
    return {
        "coord": {"lon": -0.1257, "lat": 51.5085},
        "weather": [{"id": 800, "main": "Clear", "description": "clear sky"}],
        "main": {
            "temp": 20.5,
            "feels_like": 19.8,
            "humidity": 65,
        },
        "wind": {"speed": 3.5},
    }


class TestWeatherRepositoryCreate:
    """Test creating weather API call records."""

    def test_create_call_success(self, repository, test_session, sample_weather_data):
        """Test creating a successful weather API call record."""
        call = repository.create_call(
            tenant_id="test-tenant",
            location="London",
            units="metric",
            temperature=20.5,
            feels_like=19.8,
            weather_condition="Clear",
            weather_description="clear sky",
            humidity=65,
            wind_speed=3.5,
            response_data=sample_weather_data,
            success=True,
            api_call_ms=150,
        )

        assert call.id is not None
        assert call.tenant_id == "test-tenant"
        assert call.location == "London"
        assert call.temperature == 20.5
        assert call.feels_like == 19.8
        assert call.weather_condition == "Clear"
        assert call.weather_description == "clear sky"
        assert call.humidity == 65
        assert call.wind_speed == 3.5
        assert call.success is True
        assert call.api_call_ms == 150
        assert call.created_at is not None

        # Verify it's in the database
        db_call = test_session.get(WeatherApiCall, call.id)
        assert db_call is not None
        assert db_call.location == "London"

    def test_create_call_failed(self, repository, test_session):
        """Test creating a failed weather API call record."""
        call = repository.create_call(
            tenant_id="test-tenant",
            location="InvalidCity",
            units="metric",
            temperature=None,
            feels_like=None,
            weather_condition=None,
            weather_description=None,
            humidity=None,
            wind_speed=None,
            response_data={},
            success=False,
            error_message="City not found",
            api_call_ms=None,
        )

        assert call.id is not None
        assert call.success is False
        assert call.error_message == "City not found"
        assert call.temperature is None
        assert call.api_call_ms is None

    def test_create_call_default_values(self, repository, sample_weather_data):
        """Test creating a call with default values."""
        call = repository.create_call(
            tenant_id="test-tenant",
            location="Paris",
            units="metric",
            temperature=15.0,
            feels_like=14.5,
            weather_condition="Clouds",
            weather_description="few clouds",
            humidity=70,
            wind_speed=2.5,
            response_data=sample_weather_data,
        )

        # success defaults to True
        assert call.success is True
        assert call.error_message is None
        assert call.api_call_ms is None


class TestWeatherRepositoryGetRecentCalls:
    """Test retrieving recent calls."""

    def test_get_recent_calls_empty(self, repository):
        """Test getting recent calls when none exist."""
        calls = repository.get_recent_calls("test-tenant")

        assert len(calls) == 0

    def test_get_recent_calls_single(self, repository, sample_weather_data):
        """Test getting a single recent call."""
        repository.create_call(
            tenant_id="test-tenant",
            location="London",
            units="metric",
            temperature=20.0,
            feels_like=19.0,
            weather_condition="Clear",
            weather_description="clear sky",
            humidity=60,
            wind_speed=3.0,
            response_data=sample_weather_data,
        )

        calls = repository.get_recent_calls("test-tenant")

        assert len(calls) == 1
        assert calls[0].location == "London"

    def test_get_recent_calls_multiple(self, repository, sample_weather_data):
        """Test getting multiple recent calls."""
        locations = ["London", "Paris", "Berlin", "Madrid", "Rome"]

        for location in locations:
            repository.create_call(
                tenant_id="test-tenant",
                location=location,
                units="metric",
                temperature=20.0,
                feels_like=19.0,
                weather_condition="Clear",
                weather_description="clear sky",
                humidity=60,
                wind_speed=3.0,
                response_data=sample_weather_data,
            )

        calls = repository.get_recent_calls("test-tenant")

        assert len(calls) == 5
        # Should be ordered by created_at desc (most recent first)
        assert calls[0].location == "Rome"  # Last created
        assert calls[4].location == "London"  # First created

    def test_get_recent_calls_with_limit(self, repository, sample_weather_data):
        """Test getting recent calls with limit."""
        for i in range(15):
            repository.create_call(
                tenant_id="test-tenant",
                location=f"City{i}",
                units="metric",
                temperature=20.0,
                feels_like=19.0,
                weather_condition="Clear",
                weather_description="clear sky",
                humidity=60,
                wind_speed=3.0,
                response_data=sample_weather_data,
            )

        calls = repository.get_recent_calls("test-tenant", limit=5)

        assert len(calls) == 5
        # Should get the 5 most recent
        assert calls[0].location == "City14"
        assert calls[4].location == "City10"

    def test_get_recent_calls_tenant_isolation(self, repository, sample_weather_data):
        """Test that tenant isolation works."""
        repository.create_call(
            tenant_id="tenant-1",
            location="London",
            units="metric",
            temperature=20.0,
            feels_like=19.0,
            weather_condition="Clear",
            weather_description="clear sky",
            humidity=60,
            wind_speed=3.0,
            response_data=sample_weather_data,
        )

        repository.create_call(
            tenant_id="tenant-2",
            location="Paris",
            units="metric",
            temperature=18.0,
            feels_like=17.0,
            weather_condition="Clouds",
            weather_description="few clouds",
            humidity=70,
            wind_speed=2.0,
            response_data=sample_weather_data,
        )

        tenant1_calls = repository.get_recent_calls("tenant-1")
        tenant2_calls = repository.get_recent_calls("tenant-2")

        assert len(tenant1_calls) == 1
        assert tenant1_calls[0].location == "London"

        assert len(tenant2_calls) == 1
        assert tenant2_calls[0].location == "Paris"


class TestWeatherRepositoryGetCallsByLocation:
    """Test retrieving calls by location."""

    def test_get_calls_by_location_empty(self, repository):
        """Test getting calls for location when none exist."""
        calls = repository.get_calls_by_location("test-tenant", "London")

        assert len(calls) == 0

    def test_get_calls_by_location_single(self, repository, sample_weather_data):
        """Test getting calls for a specific location."""
        repository.create_call(
            tenant_id="test-tenant",
            location="London",
            units="metric",
            temperature=20.0,
            feels_like=19.0,
            weather_condition="Clear",
            weather_description="clear sky",
            humidity=60,
            wind_speed=3.0,
            response_data=sample_weather_data,
        )

        calls = repository.get_calls_by_location("test-tenant", "London")

        assert len(calls) == 1
        assert calls[0].location == "London"

    def test_get_calls_by_location_multiple(self, repository, sample_weather_data):
        """Test getting multiple calls for same location."""
        for _ in range(3):
            repository.create_call(
                tenant_id="test-tenant",
                location="London",
                units="metric",
                temperature=20.0,
                feels_like=19.0,
                weather_condition="Clear",
                weather_description="clear sky",
                humidity=60,
                wind_speed=3.0,
                response_data=sample_weather_data,
            )

        calls = repository.get_calls_by_location("test-tenant", "London")

        assert len(calls) == 3
        for call in calls:
            assert call.location == "London"

    def test_get_calls_by_location_filters_other_locations(self, repository, sample_weather_data):
        """Test that it only returns calls for the specified location."""
        repository.create_call(
            tenant_id="test-tenant",
            location="London",
            units="metric",
            temperature=20.0,
            feels_like=19.0,
            weather_condition="Clear",
            weather_description="clear sky",
            humidity=60,
            wind_speed=3.0,
            response_data=sample_weather_data,
        )

        repository.create_call(
            tenant_id="test-tenant",
            location="Paris",
            units="metric",
            temperature=18.0,
            feels_like=17.0,
            weather_condition="Clouds",
            weather_description="few clouds",
            humidity=70,
            wind_speed=2.0,
            response_data=sample_weather_data,
        )

        london_calls = repository.get_calls_by_location("test-tenant", "London")
        paris_calls = repository.get_calls_by_location("test-tenant", "Paris")

        assert len(london_calls) == 1
        assert london_calls[0].location == "London"

        assert len(paris_calls) == 1
        assert paris_calls[0].location == "Paris"

    def test_get_calls_by_location_time_window(self, repository, test_session, sample_weather_data):
        """Test that time window filtering works."""
        # Create an old call (manually set created_at)
        old_call = WeatherApiCall(
            tenant_id="test-tenant",
            location="London",
            units="metric",
            temperature=15.0,
            feels_like=14.0,
            weather_condition="Rain",
            weather_description="light rain",
            humidity=80,
            wind_speed=5.0,
            response_data=sample_weather_data,
            created_at=datetime.utcnow() - timedelta(hours=48),
        )
        test_session.add(old_call)
        test_session.commit()

        # Create a recent call
        repository.create_call(
            tenant_id="test-tenant",
            location="London",
            units="metric",
            temperature=20.0,
            feels_like=19.0,
            weather_condition="Clear",
            weather_description="clear sky",
            humidity=60,
            wind_speed=3.0,
            response_data=sample_weather_data,
        )

        # Get calls within last 24 hours
        calls = repository.get_calls_by_location("test-tenant", "London", hours=24)

        # Should only get the recent call
        assert len(calls) == 1
        assert calls[0].weather_condition == "Clear"

        # Get calls within last 72 hours
        calls = repository.get_calls_by_location("test-tenant", "London", hours=72)

        # Should get both calls
        assert len(calls) == 2

    def test_get_calls_by_location_only_successful(self, repository, sample_weather_data):
        """Test that it only returns successful calls."""
        repository.create_call(
            tenant_id="test-tenant",
            location="London",
            units="metric",
            temperature=20.0,
            feels_like=19.0,
            weather_condition="Clear",
            weather_description="clear sky",
            humidity=60,
            wind_speed=3.0,
            response_data=sample_weather_data,
            success=True,
        )

        repository.create_call(
            tenant_id="test-tenant",
            location="London",
            units="metric",
            temperature=None,
            feels_like=None,
            weather_condition=None,
            weather_description=None,
            humidity=None,
            wind_speed=None,
            response_data={},
            success=False,
            error_message="API error",
        )

        calls = repository.get_calls_by_location("test-tenant", "London")

        # Should only get successful call
        assert len(calls) == 1
        assert calls[0].success is True


class TestWeatherRepositoryGetCallStats:
    """Test getting call statistics."""

    def test_get_call_stats_empty(self, repository):
        """Test getting stats when no calls exist."""
        stats = repository.get_call_stats("test-tenant")

        assert stats["total_calls"] == 0
        assert stats["successful_calls"] == 0
        assert stats["failed_calls"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["avg_latency_ms"] is None
        assert stats["unique_locations"] == 0

    def test_get_call_stats_single_success(self, repository, sample_weather_data):
        """Test stats with a single successful call."""
        repository.create_call(
            tenant_id="test-tenant",
            location="London",
            units="metric",
            temperature=20.0,
            feels_like=19.0,
            weather_condition="Clear",
            weather_description="clear sky",
            humidity=60,
            wind_speed=3.0,
            response_data=sample_weather_data,
            success=True,
            api_call_ms=150,
        )

        stats = repository.get_call_stats("test-tenant")

        assert stats["total_calls"] == 1
        assert stats["successful_calls"] == 1
        assert stats["failed_calls"] == 0
        assert stats["success_rate"] == 100.0
        assert stats["avg_latency_ms"] == 150.0
        assert stats["unique_locations"] == 1

    def test_get_call_stats_multiple_calls(self, repository, sample_weather_data):
        """Test stats with multiple calls."""
        # 3 successful calls
        for i in range(3):
            repository.create_call(
                tenant_id="test-tenant",
                location=f"City{i}",
                units="metric",
                temperature=20.0,
                feels_like=19.0,
                weather_condition="Clear",
                weather_description="clear sky",
                humidity=60,
                wind_speed=3.0,
                response_data=sample_weather_data,
                success=True,
                api_call_ms=100 + i * 50,  # 100, 150, 200
            )

        # 1 failed call
        repository.create_call(
            tenant_id="test-tenant",
            location="BadCity",
            units="metric",
            temperature=None,
            feels_like=None,
            weather_condition=None,
            weather_description=None,
            humidity=None,
            wind_speed=None,
            response_data={},
            success=False,
            error_message="City not found",
        )

        stats = repository.get_call_stats("test-tenant")

        assert stats["total_calls"] == 4
        assert stats["successful_calls"] == 3
        assert stats["failed_calls"] == 1
        assert stats["success_rate"] == 75.0
        assert stats["avg_latency_ms"] == 150.0  # (100 + 150 + 200) / 3
        assert stats["unique_locations"] == 4

    def test_get_call_stats_duplicate_locations(self, repository, sample_weather_data):
        """Test that unique_locations counts correctly."""
        # 3 calls to London
        for _ in range(3):
            repository.create_call(
                tenant_id="test-tenant",
                location="London",
                units="metric",
                temperature=20.0,
                feels_like=19.0,
                weather_condition="Clear",
                weather_description="clear sky",
                humidity=60,
                wind_speed=3.0,
                response_data=sample_weather_data,
                success=True,
                api_call_ms=150,
            )

        # 2 calls to Paris
        for _ in range(2):
            repository.create_call(
                tenant_id="test-tenant",
                location="Paris",
                units="metric",
                temperature=18.0,
                feels_like=17.0,
                weather_condition="Clouds",
                weather_description="few clouds",
                humidity=70,
                wind_speed=2.0,
                response_data=sample_weather_data,
                success=True,
                api_call_ms=200,
            )

        stats = repository.get_call_stats("test-tenant")

        assert stats["total_calls"] == 5
        assert stats["unique_locations"] == 2  # London and Paris

    def test_get_call_stats_time_window(self, repository, test_session, sample_weather_data):
        """Test that time window filtering works for stats."""
        # Create an old call
        old_call = WeatherApiCall(
            tenant_id="test-tenant",
            location="London",
            units="metric",
            temperature=15.0,
            feels_like=14.0,
            weather_condition="Rain",
            weather_description="light rain",
            humidity=80,
            wind_speed=5.0,
            response_data=sample_weather_data,
            success=True,
            api_call_ms=100,
            created_at=datetime.utcnow() - timedelta(days=10),
        )
        test_session.add(old_call)
        test_session.commit()

        # Create a recent call
        repository.create_call(
            tenant_id="test-tenant",
            location="Paris",
            units="metric",
            temperature=20.0,
            feels_like=19.0,
            weather_condition="Clear",
            weather_description="clear sky",
            humidity=60,
            wind_speed=3.0,
            response_data=sample_weather_data,
            success=True,
            api_call_ms=200,
        )

        # Get stats for last 7 days
        stats = repository.get_call_stats("test-tenant", days=7)

        # Should only include recent call
        assert stats["total_calls"] == 1
        assert stats["avg_latency_ms"] == 200.0

        # Get stats for last 30 days
        stats = repository.get_call_stats("test-tenant", days=30)

        # Should include both calls
        assert stats["total_calls"] == 2
        assert stats["avg_latency_ms"] == 150.0  # (100 + 200) / 2

    def test_get_call_stats_without_latency(self, repository, sample_weather_data):
        """Test stats when some calls don't have latency data."""
        repository.create_call(
            tenant_id="test-tenant",
            location="London",
            units="metric",
            temperature=20.0,
            feels_like=19.0,
            weather_condition="Clear",
            weather_description="clear sky",
            humidity=60,
            wind_speed=3.0,
            response_data=sample_weather_data,
            success=True,
            api_call_ms=150,
        )

        repository.create_call(
            tenant_id="test-tenant",
            location="Paris",
            units="metric",
            temperature=18.0,
            feels_like=17.0,
            weather_condition="Clouds",
            weather_description="few clouds",
            humidity=70,
            wind_speed=2.0,
            response_data=sample_weather_data,
            success=True,
            api_call_ms=None,  # No latency data
        )

        stats = repository.get_call_stats("test-tenant")

        assert stats["total_calls"] == 2
        assert stats["avg_latency_ms"] == 150.0  # Only counts call with latency

    def test_get_call_stats_tenant_isolation(self, repository, sample_weather_data):
        """Test that stats respect tenant isolation."""
        repository.create_call(
            tenant_id="tenant-1",
            location="London",
            units="metric",
            temperature=20.0,
            feels_like=19.0,
            weather_condition="Clear",
            weather_description="clear sky",
            humidity=60,
            wind_speed=3.0,
            response_data=sample_weather_data,
            success=True,
            api_call_ms=150,
        )

        repository.create_call(
            tenant_id="tenant-2",
            location="Paris",
            units="metric",
            temperature=18.0,
            feels_like=17.0,
            weather_condition="Clouds",
            weather_description="few clouds",
            humidity=70,
            wind_speed=2.0,
            response_data=sample_weather_data,
            success=True,
            api_call_ms=200,
        )

        tenant1_stats = repository.get_call_stats("tenant-1")
        tenant2_stats = repository.get_call_stats("tenant-2")

        assert tenant1_stats["total_calls"] == 1
        assert tenant1_stats["avg_latency_ms"] == 150.0

        assert tenant2_stats["total_calls"] == 1
        assert tenant2_stats["avg_latency_ms"] == 200.0
