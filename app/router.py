from fastapi.responses import FileResponse
from fastapi import APIRouter

router = APIRouter()


@router.get('/', tags=['default'], name='root')
async def root_path():
    return 'Hello'


@router.get('/favicon.ico', response_class=FileResponse, name='favicon', tags=['favicon'])
async def favicon() -> str:
    return 'static/images/favicon.ico'
