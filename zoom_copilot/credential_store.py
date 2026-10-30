"""Opt-in, per-user Windows Credential Manager storage; never part of app data.

Only exact provider/HTTPS endpoint targets created by this app are accessed.
There is no credential enumeration, plaintext fallback, or export operation.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import hashlib
import os
from urllib.parse import urlsplit, urlunsplit

_PREFIX = 'PrismInterview/API/v1/'
_NOT_FOUND = 1168


def credential_target(provider: str, endpoint: str) -> str:
    provider = str(provider).lower().strip()
    if provider == 'openai_compatible':
        provider = 'compatible'
    if provider not in {'qwen', 'deepseek', 'compatible'}:
        raise ValueError('该提供方不使用本机 API 密钥存储。')
    parts = urlsplit(str(endpoint).strip())
    if (parts.scheme != 'https' or not parts.hostname or parts.username or parts.password
            or parts.query or parts.fragment or parts.port not in (None, 443)):
        raise ValueError('请先填写有效的 HTTPS API 地址。')
    host = parts.hostname.lower()
    if ':' in host:
        host = '[' + host + ']'
    normalized = urlunsplit(('https', host, parts.path.rstrip('/'), '', ''))
    return _PREFIX + provider + '/' + hashlib.sha256(normalized.encode('utf-8')).hexdigest()


class _Credential(ctypes.Structure):
    _fields_ = [('Flags', wintypes.DWORD), ('Type', wintypes.DWORD),
                ('TargetName', wintypes.LPWSTR), ('Comment', wintypes.LPWSTR),
                ('LastWritten', wintypes.FILETIME), ('CredentialBlobSize', wintypes.DWORD),
                ('CredentialBlob', ctypes.POINTER(ctypes.c_ubyte)), ('Persist', wintypes.DWORD),
                ('AttributeCount', wintypes.DWORD), ('Attributes', ctypes.c_void_p),
                ('TargetAlias', wintypes.LPWSTR), ('UserName', wintypes.LPWSTR)]


def _api():
    if os.name != 'nt':
        raise OSError('本机默认密钥仅支持 Windows 凭据管理器；可继续使用会话密钥。')
    dll = ctypes.WinDLL('Advapi32.dll', use_last_error=True)
    pointer = ctypes.POINTER(_Credential)
    dll.CredWriteW.argtypes = [pointer, wintypes.DWORD]
    dll.CredWriteW.restype = wintypes.BOOL
    dll.CredReadW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(pointer)]
    dll.CredReadW.restype = wintypes.BOOL
    dll.CredDeleteW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
    dll.CredDeleteW.restype = wintypes.BOOL
    dll.CredFree.argtypes = [ctypes.c_void_p]
    dll.CredFree.restype = None
    return dll


def load_key(provider: str, endpoint: str) -> str:
    target = credential_target(provider, endpoint)
    if os.name != 'nt':
        return ''
    api = _api()
    pointer = ctypes.POINTER(_Credential)()
    if not api.CredReadW(target, 1, 0, ctypes.byref(pointer)):
        if ctypes.get_last_error() == _NOT_FOUND:
            return ''
        raise OSError('无法读取本应用的本机默认密钥，请在模型窗口重新填写。')
    try:
        return ctypes.string_at(pointer.contents.CredentialBlob,
                                pointer.contents.CredentialBlobSize).decode('utf-8')
    finally:
        api.CredFree(pointer)


def store_key(provider: str, endpoint: str, key: str) -> None:
    target = credential_target(provider, endpoint)
    data = str(key).strip().encode('utf-8')
    if not data or len(data) > 2560:
        raise ValueError('请填写有效长度的 API 密钥。')
    blob = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    value = _Credential(Type=1, TargetName=target, Comment='Prism provider and endpoint default',
                        CredentialBlobSize=len(data), CredentialBlob=blob, Persist=2,
                        UserName='PrismInterview')
    try:
        if not _api().CredWriteW(ctypes.byref(value), 0):
            raise OSError('无法保存本机默认密钥，未写入设置文件。')
    finally:
        ctypes.memset(blob, 0, len(data))


def delete_key(provider: str, endpoint: str) -> None:
    target = credential_target(provider, endpoint)
    api = _api()
    if not api.CredDeleteW(target, 1, 0) and ctypes.get_last_error() != _NOT_FOUND:
        raise OSError('无法删除此提供方/地址的本机默认密钥。')
