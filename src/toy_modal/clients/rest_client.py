"""REST-like metadata client exposed through the configured transport."""

from __future__ import annotations

from toy_modal import types
from toy_modal.futures import APIFuture


class RestClient:
    def __init__(self, *, transport, accept_tinker_paths: bool = False) -> None:
        self._transport = transport
        self.accept_tinker_paths = accept_tinker_paths

    async def _result_async(self, route: str, payload: dict, result_type):
        future = await self._transport.submit_async(route, payload, result_type=result_type)
        return await future.result_async()

    def get_training_run(
        self,
        training_run_id: types.ModelID,
        access_scope: str = "owned",
    ) -> APIFuture[types.TrainingRun]:
        return self._transport.submit(
            "rest.get_training_run",
            {"training_run_id": training_run_id, "access_scope": access_scope},
            result_type=types.TrainingRun,
        )

    async def get_training_run_async(
        self,
        training_run_id: types.ModelID,
        access_scope: str = "owned",
    ) -> types.TrainingRun:
        return await self._result_async(
            "rest.get_training_run",
            {"training_run_id": training_run_id, "access_scope": access_scope},
            types.TrainingRun,
        )

    def get_training_run_by_toy_path(
        self,
        toy_path: str,
        access_scope: str = "owned",
    ) -> APIFuture[types.TrainingRun]:
        return self._transport.submit(
            "rest.get_training_run_by_toy_path",
            {
                "toy_path": toy_path,
                "access_scope": access_scope,
                "accept_tinker_paths": self.accept_tinker_paths,
            },
            result_type=types.TrainingRun,
        )

    async def get_training_run_by_toy_path_async(
        self,
        toy_path: str,
        access_scope: str = "owned",
    ) -> types.TrainingRun:
        return await self._result_async(
            "rest.get_training_run_by_toy_path",
            {
                "toy_path": toy_path,
                "access_scope": access_scope,
                "accept_tinker_paths": self.accept_tinker_paths,
            },
            types.TrainingRun,
        )

    def get_training_run_by_tinker_path(
        self,
        tinker_path: str,
        access_scope: str = "owned",
    ) -> APIFuture[types.TrainingRun]:
        return self.get_training_run_by_toy_path(tinker_path, access_scope=access_scope)

    async def get_training_run_by_tinker_path_async(
        self,
        tinker_path: str,
        access_scope: str = "owned",
    ) -> types.TrainingRun:
        return await self.get_training_run_by_toy_path_async(
            tinker_path,
            access_scope=access_scope,
        )

    def get_weights_info_by_toy_path(self, toy_path: str) -> APIFuture[types.WeightsInfoResponse]:
        return self._transport.submit(
            "rest.get_weights_info_by_toy_path",
            {"toy_path": toy_path, "accept_tinker_paths": self.accept_tinker_paths},
            result_type=types.WeightsInfoResponse,
        )

    async def get_weights_info_by_toy_path_async(self, toy_path: str) -> types.WeightsInfoResponse:
        return await self._result_async(
            "rest.get_weights_info_by_toy_path",
            {"toy_path": toy_path, "accept_tinker_paths": self.accept_tinker_paths},
            types.WeightsInfoResponse,
        )

    def get_weights_info_by_tinker_path(self, tinker_path: str) -> APIFuture[types.WeightsInfoResponse]:
        return self.get_weights_info_by_toy_path(tinker_path)

    async def get_weights_info_by_tinker_path_async(
        self,
        tinker_path: str,
    ) -> types.WeightsInfoResponse:
        return await self.get_weights_info_by_toy_path_async(tinker_path)

    def list_training_runs(
        self,
        limit: int = 20,
        offset: int = 0,
        access_scope: str = "owned",
    ) -> APIFuture[types.TrainingRunsResponse]:
        return self._transport.submit(
            "rest.list_training_runs",
            {"limit": limit, "offset": offset, "access_scope": access_scope},
            result_type=types.TrainingRunsResponse,
        )

    async def list_training_runs_async(
        self,
        limit: int = 20,
        offset: int = 0,
        access_scope: str = "owned",
    ) -> types.TrainingRunsResponse:
        return await self._result_async(
            "rest.list_training_runs",
            {"limit": limit, "offset": offset, "access_scope": access_scope},
            types.TrainingRunsResponse,
        )

    def list_checkpoints(self, training_run_id: types.ModelID) -> APIFuture[types.CheckpointsListResponse]:
        return self._transport.submit(
            "rest.list_checkpoints",
            {"training_run_id": training_run_id},
            result_type=types.CheckpointsListResponse,
        )

    async def list_checkpoints_async(
        self,
        training_run_id: types.ModelID,
    ) -> types.CheckpointsListResponse:
        return await self._result_async(
            "rest.list_checkpoints",
            {"training_run_id": training_run_id},
            types.CheckpointsListResponse,
        )

    def list_user_checkpoints(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> APIFuture[types.CheckpointsListResponse]:
        return self._transport.submit(
            "rest.list_user_checkpoints",
            {"limit": limit, "offset": offset},
            result_type=types.CheckpointsListResponse,
        )

    async def list_user_checkpoints_async(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> types.CheckpointsListResponse:
        return await self._result_async(
            "rest.list_user_checkpoints",
            {"limit": limit, "offset": offset},
            types.CheckpointsListResponse,
        )

    def get_checkpoint_archive_url(
        self,
        training_run_id: types.ModelID,
        checkpoint_id: str,
    ) -> APIFuture[types.CheckpointArchiveUrlResponse]:
        return self._transport.submit(
            "rest.get_checkpoint_archive_url",
            {"training_run_id": training_run_id, "checkpoint_id": checkpoint_id},
            result_type=types.CheckpointArchiveUrlResponse,
        )

    async def get_checkpoint_archive_url_async(
        self,
        training_run_id: types.ModelID,
        checkpoint_id: str,
    ) -> types.CheckpointArchiveUrlResponse:
        return await self._result_async(
            "rest.get_checkpoint_archive_url",
            {"training_run_id": training_run_id, "checkpoint_id": checkpoint_id},
            types.CheckpointArchiveUrlResponse,
        )

    def get_checkpoint_archive_url_from_toy_path(
        self,
        toy_path: str,
    ) -> APIFuture[types.CheckpointArchiveUrlResponse]:
        return self._transport.submit(
            "rest.get_checkpoint_archive_url_from_toy_path",
            {"toy_path": toy_path, "accept_tinker_paths": self.accept_tinker_paths},
            result_type=types.CheckpointArchiveUrlResponse,
        )

    async def get_checkpoint_archive_url_from_toy_path_async(
        self,
        toy_path: str,
    ) -> types.CheckpointArchiveUrlResponse:
        return await self._result_async(
            "rest.get_checkpoint_archive_url_from_toy_path",
            {"toy_path": toy_path, "accept_tinker_paths": self.accept_tinker_paths},
            types.CheckpointArchiveUrlResponse,
        )

    def get_checkpoint_archive_url_from_tinker_path(
        self,
        tinker_path: str,
    ) -> APIFuture[types.CheckpointArchiveUrlResponse]:
        return self.get_checkpoint_archive_url_from_toy_path(tinker_path)

    async def get_checkpoint_archive_url_from_tinker_path_async(
        self,
        tinker_path: str,
    ) -> types.CheckpointArchiveUrlResponse:
        return await self.get_checkpoint_archive_url_from_toy_path_async(tinker_path)

    def delete_checkpoint(self, training_run_id: types.ModelID, checkpoint_id: str) -> APIFuture[None]:
        return self._transport.submit(
            "rest.delete_checkpoint",
            {"training_run_id": training_run_id, "checkpoint_id": checkpoint_id},
            result_type=None,
        )

    async def delete_checkpoint_async(
        self,
        training_run_id: types.ModelID,
        checkpoint_id: str,
    ) -> None:
        return await self._result_async(
            "rest.delete_checkpoint",
            {"training_run_id": training_run_id, "checkpoint_id": checkpoint_id},
            None,
        )

    def delete_checkpoint_from_toy_path(self, toy_path: str) -> APIFuture[None]:
        return self._transport.submit(
            "rest.delete_checkpoint_from_toy_path",
            {"toy_path": toy_path, "accept_tinker_paths": self.accept_tinker_paths},
            result_type=None,
        )

    async def delete_checkpoint_from_toy_path_async(self, toy_path: str) -> None:
        return await self._result_async(
            "rest.delete_checkpoint_from_toy_path",
            {"toy_path": toy_path, "accept_tinker_paths": self.accept_tinker_paths},
            None,
        )

    def delete_checkpoint_from_tinker_path(self, tinker_path: str) -> APIFuture[None]:
        return self.delete_checkpoint_from_toy_path(tinker_path)

    async def delete_checkpoint_from_tinker_path_async(self, tinker_path: str) -> None:
        return await self.delete_checkpoint_from_toy_path_async(tinker_path)

    def set_checkpoint_ttl_from_toy_path(
        self,
        toy_path: str,
        ttl_seconds: int | None,
    ) -> APIFuture[None]:
        return self._transport.submit(
            "rest.set_checkpoint_ttl_from_toy_path",
            {
                "toy_path": toy_path,
                "ttl_seconds": ttl_seconds,
                "accept_tinker_paths": self.accept_tinker_paths,
            },
            result_type=None,
        )

    async def set_checkpoint_ttl_from_toy_path_async(
        self,
        toy_path: str,
        ttl_seconds: int | None,
    ) -> None:
        return await self._result_async(
            "rest.set_checkpoint_ttl_from_toy_path",
            {
                "toy_path": toy_path,
                "ttl_seconds": ttl_seconds,
                "accept_tinker_paths": self.accept_tinker_paths,
            },
            None,
        )

    def set_checkpoint_ttl_from_tinker_path(
        self,
        tinker_path: str,
        ttl_seconds: int | None,
    ) -> APIFuture[None]:
        return self.set_checkpoint_ttl_from_toy_path(tinker_path, ttl_seconds)

    async def set_checkpoint_ttl_from_tinker_path_async(
        self,
        tinker_path: str,
        ttl_seconds: int | None,
    ) -> None:
        return await self.set_checkpoint_ttl_from_toy_path_async(tinker_path, ttl_seconds)

    def publish_checkpoint_from_toy_path(self, toy_path: str) -> APIFuture[None]:
        return self._transport.submit(
            "rest.set_checkpoint_public",
            {
                "toy_path": toy_path,
                "public": True,
                "accept_tinker_paths": self.accept_tinker_paths,
            },
            result_type=None,
        )

    async def publish_checkpoint_from_toy_path_async(self, toy_path: str) -> None:
        return await self._result_async(
            "rest.set_checkpoint_public",
            {
                "toy_path": toy_path,
                "public": True,
                "accept_tinker_paths": self.accept_tinker_paths,
            },
            None,
        )

    def publish_checkpoint_from_tinker_path(self, tinker_path: str) -> APIFuture[None]:
        return self.publish_checkpoint_from_toy_path(tinker_path)

    async def publish_checkpoint_from_tinker_path_async(self, tinker_path: str) -> None:
        return await self.publish_checkpoint_from_toy_path_async(tinker_path)

    def unpublish_checkpoint_from_toy_path(self, toy_path: str) -> APIFuture[None]:
        return self._transport.submit(
            "rest.set_checkpoint_public",
            {
                "toy_path": toy_path,
                "public": False,
                "accept_tinker_paths": self.accept_tinker_paths,
            },
            result_type=None,
        )

    async def unpublish_checkpoint_from_toy_path_async(self, toy_path: str) -> None:
        return await self._result_async(
            "rest.set_checkpoint_public",
            {
                "toy_path": toy_path,
                "public": False,
                "accept_tinker_paths": self.accept_tinker_paths,
            },
            None,
        )

    def unpublish_checkpoint_from_tinker_path(self, tinker_path: str) -> APIFuture[None]:
        return self.unpublish_checkpoint_from_toy_path(tinker_path)

    async def unpublish_checkpoint_from_tinker_path_async(self, tinker_path: str) -> None:
        return await self.unpublish_checkpoint_from_toy_path_async(tinker_path)

    def get_session(
        self,
        session_id: str,
        access_scope: str = "owned",
    ) -> APIFuture[types.GetSessionResponse]:
        return self._transport.submit(
            "rest.get_session",
            {"session_id": session_id, "access_scope": access_scope},
            result_type=types.GetSessionResponse,
        )

    async def get_session_async(
        self,
        session_id: str,
        access_scope: str = "owned",
    ) -> types.GetSessionResponse:
        return await self._result_async(
            "rest.get_session",
            {"session_id": session_id, "access_scope": access_scope},
            types.GetSessionResponse,
        )

    def list_sessions(
        self,
        limit: int = 20,
        offset: int = 0,
        access_scope: str = "owned",
    ) -> APIFuture[types.ListSessionsResponse]:
        return self._transport.submit(
            "rest.list_sessions",
            {"limit": limit, "offset": offset, "access_scope": access_scope},
            result_type=types.ListSessionsResponse,
        )

    async def list_sessions_async(
        self,
        limit: int = 20,
        offset: int = 0,
        access_scope: str = "owned",
    ) -> types.ListSessionsResponse:
        return await self._result_async(
            "rest.list_sessions",
            {"limit": limit, "offset": offset, "access_scope": access_scope},
            types.ListSessionsResponse,
        )

    def get_sampler(self, sampler_id: str) -> APIFuture[types.GetSamplerResponse]:
        return self._transport.submit(
            "rest.get_sampler",
            {"sampler_id": sampler_id},
            result_type=types.GetSamplerResponse,
        )

    async def get_sampler_async(self, sampler_id: str) -> types.GetSamplerResponse:
        return await self._result_async(
            "rest.get_sampler",
            {"sampler_id": sampler_id},
            types.GetSamplerResponse,
        )
