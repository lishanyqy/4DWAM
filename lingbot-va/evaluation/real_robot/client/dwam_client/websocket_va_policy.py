"""WebSocket client for LingBot-VA ``VA_Server`` protocol.

Protocol (msgpack + numpy):
  - reset:  {reset: True, prompt: str} -> {}
  - infer:  {obs: frame|list} -> {action: ndarray (C, F, H), ...}
  - kv:     {compute_kv_cache: True, obs: list, state: action} -> {}

Connection defaults match openpi-on-LIFT2 habits: port 7777, no compression,
``ping_interval=None`` so long diffusion steps do not time out.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import websockets.sync.client

from .base_policy import BasePolicy
from .msgpack_numpy import Packer, unpackb

logger = logging.getLogger(__name__)

ObsFrame = Dict[str, Any]
ObsInput = Union[ObsFrame, Sequence[ObsFrame]]


class WebsocketVAPolicy(BasePolicy):
    """Client for ``wan_va.utils.Simple_Remote_Infer`` WebsocketPolicyServer."""

    def __init__(
        self,
        host: str = '127.0.0.1',
        port: Optional[int] = 7777,
        api_key: Optional[str] = None,
        connect_retry_seconds: float = 5.0,
        connect_timeout_seconds: float = 10.0,
        metadata_timeout_seconds: float = 10.0,
        max_wait_seconds: Optional[float] = None,
    ) -> None:
        if host.startswith('ws://') or host.startswith('wss://'):
            # Full URI provided. Only append port when the URI has no explicit port.
            # Examples:
            #   ws://192.168.1.1        + port 7777 -> ws://192.168.1.1:7777
            #   ws://192.168.1.1:9000   + port 7777 -> unchanged
            self._uri = host.rstrip('/')
            has_explicit_port = False
            try:
                # strip scheme
                without_scheme = self._uri.split('://', 1)[1]
                # strip path
                host_port = without_scheme.split('/', 1)[0]
                if host_port.startswith('['):
                    # IPv6 ws://[::1]:port
                    has_explicit_port = ']:' in host_port
                else:
                    has_explicit_port = host_port.count(':') == 1
            except Exception:  # noqa: BLE001
                has_explicit_port = True
            if port is not None and not has_explicit_port:
                self._uri = f'{self._uri}:{int(port)}'
        else:
            self._uri = f'ws://{host}'
            if port is not None:
                self._uri += f':{int(port)}'

        self._api_key = api_key
        self._connect_retry_seconds = float(connect_retry_seconds)
        self._connect_timeout_seconds = float(connect_timeout_seconds)
        self._metadata_timeout_seconds = float(metadata_timeout_seconds)
        self._max_wait_seconds = (
            None
            if max_wait_seconds is None or float(max_wait_seconds) <= 0
            else float(max_wait_seconds)
        )
        if self._connect_retry_seconds < 0:
            raise ValueError('connect_retry_seconds must be non-negative')
        if self._connect_timeout_seconds <= 0:
            raise ValueError('connect_timeout_seconds must be positive')
        if self._metadata_timeout_seconds <= 0:
            raise ValueError('metadata_timeout_seconds must be positive')
        self._packer = Packer()
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict[str, Any]:
        return dict(self._server_metadata or {})

    def _wait_for_server(
        self,
    ) -> Tuple[websockets.sync.client.ClientConnection, Dict[str, Any]]:
        logger.info('Waiting for VA server at %s ...', self._uri)
        wait_started_at = time.monotonic()
        attempt_index = 0

        while True:
            attempt_index += 1
            connection = None
            try:
                headers = (
                    {'Authorization': f'Api-Key {self._api_key}'}
                    if self._api_key
                    else None
                )
                logger.info(
                    'WebSocket connection attempt %d to %s',
                    attempt_index,
                    self._uri,
                )
                connection = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    proxy=None,
                    open_timeout=self._connect_timeout_seconds,
                    ping_interval=None,
                    close_timeout=10,
                )
                metadata = unpackb(
                    connection.recv(timeout=self._metadata_timeout_seconds)
                )
                if not isinstance(metadata, dict):
                    metadata = {}
                logger.info('Connected to VA server at %s', self._uri)
                return connection, metadata
            except KeyboardInterrupt:
                if connection is not None:
                    connection.close()
                logger.info('Interrupted while connecting to VA server')
                raise
            except Exception as error:  # noqa: BLE001 - reconnect loop
                if connection is not None:
                    try:
                        connection.close()
                    except Exception:  # noqa: BLE001 - best-effort cleanup
                        pass

                elapsed_seconds = time.monotonic() - wait_started_at
                if (
                    self._max_wait_seconds is not None
                    and elapsed_seconds >= self._max_wait_seconds
                ):
                    raise TimeoutError(
                        f'Timed out after {elapsed_seconds:.1f}s waiting for '
                        f'VA server at {self._uri}; last error: {error}'
                    ) from error

                retry_seconds = self._connect_retry_seconds
                if self._max_wait_seconds is not None:
                    remaining_seconds = self._max_wait_seconds - elapsed_seconds
                    retry_seconds = min(retry_seconds, max(remaining_seconds, 0.0))

                logger.warning(
                    'VA server connection attempt %d failed for %s: %s; '
                    'retrying in %.1fs',
                    attempt_index,
                    self._uri,
                    error,
                    retry_seconds,
                )
                if retry_seconds <= 0:
                    raise TimeoutError(
                        f'Timed out waiting for VA server at {self._uri}'
                    ) from error
                time.sleep(retry_seconds)

    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._ws.send(self._packer.pack(payload))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f'Error in VA inference server:\n{response}')
        result = unpackb(response)
        if not isinstance(result, dict):
            raise RuntimeError(f'Unexpected server response type: {type(result)}')
        return result

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Low-level request: send a fully-formed payload dict."""
        return self._request(obs)

    def reset(self, prompt: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {'reset': True}
        if prompt is not None:
            payload['prompt'] = prompt
        return self._request(payload)

    def infer_action(self, obs: ObsInput, **extra: Any) -> Dict[str, Any]:
        """Request one action chunk. ``obs`` is a frame dict or list of frames."""
        payload: Dict[str, Any] = {'obs': obs}
        payload.update(extra)
        result = self._request(payload)
        if 'action' not in result:
            raise KeyError(
                f"VA server response missing 'action' key; got keys={list(result.keys())}"
            )
        return result

    def compute_kv_cache(
        self,
        obs: ObsInput,
        state: Any,
        **extra: Any,
    ) -> Dict[str, Any]:
        """Write real observations + action state into the server KV cache."""
        payload: Dict[str, Any] = {
            'compute_kv_cache': True,
            'obs': obs,
            'state': state,
        }
        payload.update(extra)
        return self._request(payload)

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:  # noqa: BLE001
            pass
