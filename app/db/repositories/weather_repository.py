"""Weather API call repository for database operations."""

from datetime import datetime, timedelta
from typing import Sequence

from sqlmodel import Session, func, select

from app.db.models import WeatherApiCall


class WeatherRepository:
    """Repository for weather API call database operations.

    Provides CRUD operations and specialized queries for weather API call history.
    Supports caching, auditing, and statistics tracking.
    """

    def __init__(self, session: Session):
        """Initialize repository with database session.

        Args:
            session: SQLModel database session
        """
        self.session = session

    def create_call(
        self,
        tenant_id: str,
        location: str,
        units: str,
        temperature: float | None,
        feels_like: float | None,
        weather_condition: str | None,
        weather_description: str | None,
        humidity: int | None,
        wind_speed: float | None,
        response_data: dict,
        success: bool = True,
        error_message: str | None = None,
        api_call_ms: int | None = None,
    ) -> WeatherApiCall:
        """Create a weather API call record.

        Args:
            tenant_id: Tenant identifier
            location: City name or coordinates
            units: Temperature units (metric, imperial, standard)
            temperature: Temperature value
            feels_like: Feels-like temperature
            weather_condition: Weather condition (e.g., "Clear")
            weather_description: Detailed description (e.g., "clear sky")
            humidity: Humidity percentage
            wind_speed: Wind speed value
            response_data: Full API response as dict
            success: Whether the API call succeeded
            error_message: Error message if call failed
            api_call_ms: API call latency in milliseconds

        Returns:
            WeatherApiCall: Created database record
        """
        call = WeatherApiCall(
            tenant_id=tenant_id,
            location=location,
            units=units,
            temperature=temperature,
            feels_like=feels_like,
            weather_condition=weather_condition,
            weather_description=weather_description,
            humidity=humidity,
            wind_speed=wind_speed,
            response_data=response_data,
            success=success,
            error_message=error_message,
            api_call_ms=api_call_ms,
        )

        self.session.add(call)
        self.session.commit()
        self.session.refresh(call)

        return call

    def get_recent_calls(self, tenant_id: str, limit: int = 10) -> Sequence[WeatherApiCall]:
        """Get recent weather API calls for a tenant.

        Args:
            tenant_id: Tenant identifier
            limit: Maximum number of records to return

        Returns:
            Sequence[WeatherApiCall]: List of recent calls, most recent first
        """
        statement = (
            select(WeatherApiCall)
            .where(WeatherApiCall.tenant_id == tenant_id)
            .order_by(WeatherApiCall.created_at.desc())
            .limit(limit)
        )

        results = self.session.exec(statement)
        return results.all()

    def get_calls_by_location(
        self, tenant_id: str, location: str, hours: int = 24
    ) -> Sequence[WeatherApiCall]:
        """Get recent calls for a specific location (for caching).

        Args:
            tenant_id: Tenant identifier
            location: City name or coordinates
            hours: Number of hours to look back

        Returns:
            Sequence[WeatherApiCall]: List of calls for the location within time window
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        statement = (
            select(WeatherApiCall)
            .where(
                WeatherApiCall.tenant_id == tenant_id,
                WeatherApiCall.location == location,
                WeatherApiCall.created_at >= cutoff_time,
                WeatherApiCall.success == True,  # noqa: E712
            )
            .order_by(WeatherApiCall.created_at.desc())
        )

        results = self.session.exec(statement)
        return results.all()

    def get_call_stats(self, tenant_id: str, days: int = 7) -> dict:
        """Get statistics about API calls.

        Calculates total calls, success rate, and average latency for the specified
        time period.

        Args:
            tenant_id: Tenant identifier
            days: Number of days to include in statistics

        Returns:
            dict: Statistics including:
                - total_calls: Total number of API calls
                - successful_calls: Number of successful calls
                - failed_calls: Number of failed calls
                - success_rate: Percentage of successful calls
                - avg_latency_ms: Average API latency in milliseconds
                - unique_locations: Number of unique locations queried
        """
        cutoff_time = datetime.utcnow() - timedelta(days=days)

        # Get total calls
        total_statement = select(func.count(WeatherApiCall.id)).where(
            WeatherApiCall.tenant_id == tenant_id,
            WeatherApiCall.created_at >= cutoff_time,
        )
        total_calls = self.session.exec(total_statement).one()

        # Get successful calls
        success_statement = select(func.count(WeatherApiCall.id)).where(
            WeatherApiCall.tenant_id == tenant_id,
            WeatherApiCall.created_at >= cutoff_time,
            WeatherApiCall.success == True,  # noqa: E712
        )
        successful_calls = self.session.exec(success_statement).one()

        # Get average latency (only for successful calls with latency data)
        avg_latency_statement = select(func.avg(WeatherApiCall.api_call_ms)).where(
            WeatherApiCall.tenant_id == tenant_id,
            WeatherApiCall.created_at >= cutoff_time,
            WeatherApiCall.success == True,  # noqa: E712
            WeatherApiCall.api_call_ms.is_not(None),
        )
        avg_latency = self.session.exec(avg_latency_statement).one()

        # Get unique locations
        unique_locations_statement = select(
            func.count(func.distinct(WeatherApiCall.location))
        ).where(
            WeatherApiCall.tenant_id == tenant_id,
            WeatherApiCall.created_at >= cutoff_time,
        )
        unique_locations = self.session.exec(unique_locations_statement).one()

        failed_calls = total_calls - successful_calls
        success_rate = (successful_calls / total_calls * 100) if total_calls > 0 else 0.0

        return {
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": failed_calls,
            "success_rate": round(success_rate, 2),
            "avg_latency_ms": round(avg_latency, 2) if avg_latency else None,
            "unique_locations": unique_locations,
        }
