import uvicorn
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
from fastapi.staticfiles import StaticFiles

from app.conf.config import settings
from .router import router
from .api import api


def custom_generate_unique_id(route: APIRoute):
    return route.name


def get_application(
        root_path: Optional[str] = None,
        root_path_in_servers: Optional[bool] = False,
        openapi_url: Optional[str] = "/openapi.json",
):
    application = FastAPI(
        title=settings.PROJECT_NAME,
        debug=settings.DEBUG,
        version=settings.VERSION,
        openapi_url=openapi_url,
        root_path=root_path,
        root_path_in_servers=root_path_in_servers,
        generate_unique_id_function=custom_generate_unique_id
    )

    # Set all CORS enabled origins
    if settings.BACKEND_CORS_ORIGINS:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    else:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=['*'],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    application.mount("/static", StaticFiles(directory="static", html=True), name="static")
    application.mount("/media", StaticFiles(directory="media", html=True), name="media")
    application.include_router(api, prefix="api")
    application.include_router(router)

    return application


app = get_application(
    root_path=settings.ROOT_PATH,
    root_path_in_servers=settings.ROOT_PATH_IN_SERVERS,
    openapi_url=settings.OPENAPI_URL,
)

if __name__ == "__main__":
    # noinspection PyTypeChecker
    uvicorn.run(
        get_application(
            root_path=settings.ROOT_PATH,
            root_path_in_servers=settings.ROOT_PATH_IN_SERVERS,
            openapi_url=settings.OPENAPI_URL,
        ),
        host="0.0.0.0", port=8000
    )
