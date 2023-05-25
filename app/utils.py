import os
from typing import Optional
from datetime import datetime


def upload_to_dir(
        file_dir: Optional[str] = '',
        with_media: Optional[bool] = False,
):
    now: datetime = datetime.now()
    parent_dir: str = f'{file_dir}/{now:%Y/%m/%d}'
    if with_media:
        return f'media/{parent_dir}'
    return parent_dir


def upload_to(filename: str, extension: str, file_dir: Optional[str] = '', ) -> str:
    """
    Return path seperated by date, starts with slash
    :param filename:
    :param extension:
    :param file_dir:
    :return:
    """
    now: datetime = datetime.now()
    extension: str = extension.lower()
    parent_dir: str = f'{file_dir}/{now:%Y/%m/%d}'
    os.makedirs(f'media/{parent_dir}', mode=0o777, exist_ok=True)

    return f"{parent_dir}/{filename}{extension}"
