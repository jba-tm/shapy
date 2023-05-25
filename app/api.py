import os
import sys

import torch
import smplx
import trimesh

from enum import Enum
from uuid import uuid4
from typing import Literal, Optional, List
from fastapi import APIRouter
from pydantic import BaseModel, Field
from PIL import ImageDraw, ImageFont
from omegaconf import OmegaConf
from loguru import logger

# Custom compiles
from body_measurements import BodyMeasurements
from attributes.utils.renderer import Renderer
from attributes.utils.config import default_conf
from attributes.attributes_betas.build import MODEL_DICT
from attributes.dataloader.demo import DEMO_S2A, DEMO_A2S

from app.conf.config import settings
from .utils import upload_to, upload_to_dir

api = APIRouter()


class MeasurementIn(BaseModel):
    age: Optional[Literal["adult", "kid"]] = "adult"
    data: List[float] = Field(..., description="numpy data")
    num_betas: Optional[int] = Field(10, description="number of betas smpl model uses")
    gender: Optional[Literal["male", "female", "neutral"]] = Field("neutral", description="gender of smpl model")
    render: Optional[bool] = Field(False)


class AttributeConfigEnum(Enum):
    ZERO_ZERO_A2S = "00_a2s"
    ZERO_ONE_A_H2S = "00_h2s"
    ZERO_ONE_B_AH2S = "01b_ah2s"
    ZERO_TWO_A_HW2S = "02a_hw2s"
    ZERO_TWO_B_AHW2S = "02b_ahw2s"
    ZERO_THREE_A_C2S = "03a_c2s"
    ZERO_THREE_B_AC2S = "03b_ac2s"
    ZERO_FOUR_A_HCWH2S = "04a_hcwh2s"
    ZERO_FOUR_B_AHCWH2S = "04b_ahcwh2s"
    ZERO_FIVE_A_HWCWH2S = "05a_hwcwh2s"
    ZERO_FIVE_B_AHWCWH2S = "05b_ahwcwh2s"


class AttributesIn(BaseModel):
    exp_cfgs: Optional[List[AttributeConfigEnum]] = []
    # exp_opts: Optional[list] = Field([], description="The configuration of the Detector")
    network_type: Optional[Literal["a2b", "b2a"]] = "a2b"
    ds_gender: Optional[str] = ""
    model_gender: Optional[str] = ""
    model_type: Optional[Literal["smplx"]] = "smplx"
    render: Optional[bool] = Field(False)
    num_betas: Optional[int] = Field(10, description="number of betas smpl model uses")


@api.post('/attributes/', name='attributes', )
def get_attributes(
        obj_in: AttributesIn
):
    cfg = default_conf.copy()
    for exp_cfg in obj_in.exp_cfgs:
        if exp_cfg:
            cfg.merge_with(OmegaConf.load(exp_cfg.value))
    # if cmd_args.exp_opts:
    cfg.merge_with({
        "type": obj_in.network_type,
        "ds_gender": obj_in.ds_gender,
        "model_gender": obj_in.model_gender,
        "model_type": obj_in.model_type,
    })

    n_type = cfg.get('type')
    cfg_ds_gender = cfg.get('ds_gender', '')
    cfg_model_gender = cfg.get('model_gender', '')
    cfg_model_type = cfg.get('model_type', 'smplx')
    checkout_name = uuid4().hex
    checkpoint_path = upload_to(filename=checkout_name, extension='ckpt', file_dir='attributes')
    result = None
    if n_type == 'a2b':
        loaded_model = MODEL_DICT[n_type].load_from_checkpoint(
            checkpoint_path=checkpoint_path
        )

        dataset = DEMO_A2S(
            ds_gender=cfg_ds_gender,
            model_gender=cfg_model_gender,
            model_type=cfg_model_type,
            rating_folder='../samples/attributes/'
        )

        test_input, _ = loaded_model.create_input_feature_vec(dataset.db)

        test_input = loaded_model.to_whw2s(test_input, None) if loaded_model.whw2s_model else test_input
        prediction = loaded_model.a2b.predict(test_input)
        data = {}
        for idx, betas in enumerate(prediction):
            model_name = dataset.db['ids'][idx]
            data[model_name] = betas.detach().cpu().numpy()
        result = {
            'data': data,
            'images': None,
        }
        if obj_in.render:
            import trimesh
            import torch
            import smplx
            import sys
            from attributes.utils.renderer import Renderer
            renderer = Renderer(
                is_registration=False
            )

            device = torch.device('cuda')
            if not torch.cuda.is_available():
                logger.error('CUDA is not available!')
                sys.exit(3)

            smpl = smplx.create(
                model_path=settings.SMPL_MODEL_PATH,
                gender=cfg_model_gender,
                num_betas=obj_in.num_betas,
                model_type=cfg_model_type,

            ).to(device)
            result_images = []
            for idx, betas in enumerate(prediction):
                body = smpl(betas=betas.unsqueeze(0).to(device))
                shaped_vertices = body['v_shaped']
                pred_mesh = trimesh.Trimesh(shaped_vertices.detach().cpu().numpy()[0], smpl.faces)
                pred_img = renderer.render(pred_mesh)
                filename = dataset.db['ids'][idx]
                path = upload_to(filename=filename, extension='png', file_dir='attributes')
                pred_img.save(path)
                result_images.append(path)
            result['images'] = result_images
    # elif n_type == 'b2a':
    #     loaded_model = MODEL_DICT[n_type].load_from_checkpoint(
    #         checkpoint_path=checkpoint_path)
    #
    #     dataset = DEMO_S2A(
    #         betas_folder='../samples/shapy_fit/',
    #         ds_genders_path='../samples/genders.yaml',
    #         model_gender=cfg_model_gender,
    #         model_type=cfg_model_type,
    #     )

    return result


@api.post('/regressor/', name='regressor')
def get_regressor():
    return {}


@api.post("/measurements/", name='measurements')
async def get_measurements(
        obj_in: MeasurementIn,
):
    device = torch.device('cuda')

    body_measurements = BodyMeasurements(
        {
            'meas_definition_path': settings.MEAS_DEFINITION_PATH,
            'meas_vertices_path': settings.MEAS_VERTICES_PATH
        },
    ).to(device)
    smpl = smplx.create(
        model_path=settings.SMPL_MODEL_PATH,
        gender=obj_in.gender,
        num_betas=obj_in.num_betas,
        model_type='smplx',
        age=obj_in.age,
    ).to(device)

    betas = torch.from_numpy(obj_in.data).to(device).unsqueeze(0)

    body = smpl(betas=betas)
    shaped_vertices = body['v_shaped']
    shaped_triangles = shaped_vertices[:, smpl.faces_tensor]

    # Compute the measurements on the body
    measurements = body_measurements(shaped_triangles)['measurements']
    result = {

        "measurements": measurements,
        "image": None
    }
    # add measurements to image and save image
    if obj_in.render:
        renderer = Renderer(
            is_registration=False
        )
        pred_mesh = trimesh.Trimesh(shaped_vertices.cpu().numpy()[0], smpl.faces)
        pred_img = renderer.render(pred_mesh)

        # print result
        mmts_str = '    Virtual measurements: '
        for k, v in measurements.items():
            value = v['tensor'].item()
            unit = 'kg' if k == 'mass' else 'm'
            mmts_str += f'    {k}: {value:.2f} {unit}'

        font = ImageFont.truetype("../samples/OpenSans-Regular.ttf", size=24)
        ImageDraw.Draw(pred_img).text(
            (0, 10), mmts_str, (0, 0, 0), font=font
        )
        img_name = uuid4().hex
        path = upload_to(img_name, 'png', "measurements")
        pred_img.save(path)
        result['image'] = path
    return result


@api.get('/about/')
def show_about():
    """
    Get deployment information, for debugging
    """

    def bash(command):
        output = os.popen(command).read()
        return output

    return {
        "sys.version": sys.version,
        "torch.__version__": torch.__version__,
        "torch.cuda.is_available()": torch.cuda.is_available(),
        "torch.version.cuda": torch.version.cuda,
        "torch.backends.cudnn.version()": torch.backends.cudnn.version(),
        "torch.backends.cudnn.enabled": torch.backends.cudnn.enabled,
        "nvidia-smi": bash('nvidia-smi')
    }
