from abc import ABC, abstractmethod
from app.domain.jobs.models.raw_job import RawJob


class JobSource(ABC):
    """Abstract Base Class (interface) representing an external job source provider.

    All external job feed implementations (e.g. Arbeitnow, RemoteOK, LinkedIn) must implement
    this interface to keep business services decoupled from specific API endpoints.
    """

    @abstractmethod
    async def fetch_jobs(
        self,
        query: str,
        location: str | None = None,
        page: int = 1,
        limit: int = 20,
    ) -> list[RawJob]:
        """Fetch raw job postings from the external provider based on query parameters.

        Args:
            query (str): Job search query (e.g. title, keywords).
            location (str | None): Optional location filter.
            page (int): Page number for pagination. Defaults to 1.
            limit (int): Number of jobs to return per page. Defaults to 20.

        Returns:
            list[RawJob]: A list of raw job postings.
        """
        pass

    @abstractmethod
    async def health(self) -> bool:
        """Perform a connection health check to the job source provider.

        Returns:
            bool: True if the provider is reachable and healthy, False otherwise.
        """
        pass
