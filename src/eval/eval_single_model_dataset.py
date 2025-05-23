import pdb
import time
from glob import glob
import argparse
from rich.table import Table
from typing import Tuple, Dict
from rich.progress import track
import json
import math
import tqdm
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2

from src.models.components.kappamask import KappaMask
from src.models.components.dbnet import DBNet
from src.models.components.unetmobv2 import UNetMobV2
from src.models.components.unet import UNet

from collections import OrderedDict
from src.models.components.sctnet import SCTNet
from src.metrics.metric import IoUMetric
from torchmetrics.utilities.data import to_onehot
import albumentations as albu
import torch
from torch import nn as nn
from torch.nn import functional as F
from src.data.hrc_whu_datamodule import HRC_WHU
from src.data.hrc_whu_datamodule import HRC_WHUDataModule
from src.data.cloudsen12_high_datamodule import CloudSEN12HighDataModule
from src.data.gf12ms_whu_datamodule import GF12MSWHUDataModule
from src.data.l8_biome_crop_datamodule import L8BiomeCropDataModule
import pickle


def get_args():
    parser = argparse.ArgumentParser(description="获取实验名称和使用的显卡信息")
    parser.add_argument(
        "--dataset_name", type=str, help="数据集名称", default="cloudsen12_high_l1c"
    )
    parser.add_argument(
        "--model_name", type=str, help="模型名称", default="sam"
    )
    parser.add_argument(
        "--model_path", type=str, help="权重", default="cloudsen12_high_l1c"
    )
    parser.add_argument("--gpu", type=str, help="使用的设备", default="cuda:0")

    args = parser.parse_args()
    return args.dataset_name, args.model_name,args.model_path,args.gpu


class Eval:
    def __init__(self, dataset_name: str, model_name:str,model_path:str,device: str):
        self.device = device
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.num_classes,self.image_size,self.colors = self.__get_num_classes_image_shape_colors(dataset_name)
        self.root = self.__get_root(self.dataset_name)

        self.weight_path = model_path
        self.model = SCTNet(num_classes=self.num_classes).to(self.device)
        # self.model = UNet(num_classes=self.num_classes).to(self.device)
        # self.model = UNetMobV2(num_classes=self.num_classes).to(self.device)
        # self.model = DBNet(img_size=256, in_channels=3, num_classes=self.num_classes).to(self.device)
        # self.model = KappaMask(in_channels=3, num_classes=self.num_classes).to(self.device)

        self.__load_weight()
        self.val_dataloader = self.__load_data(dataset_name=self.dataset_name)



    def __get_num_classes_image_shape_colors(self, experiment_name: str):
        if experiment_name in ["cloudsen12_high_l1c", "cloudsen12_high_l2a"]:
            return (
                4,
                512,
                (
                    (0, 0, 0),
                    (255, 255, 255),
                    (170, 170, 170),
                    (85, 85, 85),
                ),
            )
        elif experiment_name in ["gf12ms_whu_gf1", "gf12ms_whu_gf2"]:
            return 2, 256, ((0, 0, 0), (255, 255, 255))
        elif experiment_name in ["hrc_whu"]:
            return 2, 256, ((0, 0, 0), (255, 255, 255))
        elif experiment_name in ["l8_biome_crop"]:
            return (
                4,
                512,
                (
                    (0, 0, 0),
                    (85, 85, 85),
                    (170, 170, 170),
                    (255, 255, 255),
                ),
            )
        raise ValueError(f"Experiment name {experiment_name} is not recognized.")

    def __get_root(self, dataset_name: str):
        dataset_root_mapping = {
            "cloudsen12_high_l1c": "data/cloudsen12_high",
            "cloudsen12_high_l2a": "data/cloudsen12_high",
            "gf12ms_whu_gf1": "data/gf12ms_whu",
            "gf12ms_whu_gf2": "data/gf12ms_whu",
            "hrc_whu": "data/hrc_whu",
            "l8_biome_crop": "data/l8_biome_crop",
        }
        return dataset_root_mapping[dataset_name]

    def __load_weight(self):
        """
        将模型权重加载进来
        """

        weight = torch.load(self.weight_path, map_location=self.device)
        state_dict = {}
        for key, value in weight["state_dict"].items():
            new_key = key[4:]
            state_dict[new_key] = value
        self.model.load_state_dict(state_dict,strict=False)
        self.model.eval()

    def __load_data(self, dataset_name: str):

        data_loader = self.__get_data_module(dataset_name)
        data_loader.prepare_data()
        if dataset_name == "l8_biome_crop":

            data_loader.setup("test")
        else:
            data_loader.setup()
        val_dataloader = data_loader.test_dataloader()
        return val_dataloader

    def __get_data_module(self, experiment_name):
        train_pipeline = val_pipeline = test_pipeline = dict(
            all_transform=albu.Compose(
                [
                    albu.PadIfNeeded(
                        self.image_size, self.image_size, p=1, always_apply=True
                    ),
                    albu.CenterCrop(self.image_size, self.image_size),
                ]
            ),
            img_transform=albu.Compose([ToTensorV2()]),
            ann_transform=None,
        )
        if experiment_name == "cloudsen12_high_l1c":
            return CloudSEN12HighDataModule(
                root=self.root,
                level="l1c",
                train_pipeline=train_pipeline,
                val_pipeline=val_pipeline,
                test_pipeline=test_pipeline,
                batch_size=1,
            )

        elif experiment_name == "cloudsen12_high_l2a":
            return CloudSEN12HighDataModule(
                root=self.root,
                level="l2a",
                train_pipeline=train_pipeline,
                val_pipeline=val_pipeline,
                test_pipeline=test_pipeline,
                batch_size=1,
            )
        elif experiment_name == "gf12ms_whu_gf1":
            return GF12MSWHUDataModule(
                root=self.root,
                train_pipeline=train_pipeline,
                val_pipeline=val_pipeline,
                test_pipeline=test_pipeline,
                batch_size=1,
                serial="gf1",
            )
        elif experiment_name == "gf12ms_whu_gf2":
            return GF12MSWHUDataModule(
                root=self.root,
                train_pipeline=train_pipeline,
                val_pipeline=val_pipeline,
                test_pipeline=test_pipeline,
                batch_size=1,
                # batch_size=64,
                serial="gf2",
            )
        elif experiment_name == "hrc_whu":
            train_pipeline = val_pipeline = test_pipeline = dict(
                all_transform=albu.Compose(
                    [albu.CenterCrop(self.image_size, self.image_size)]
                ),
                img_transform=albu.Compose([albu.ToFloat(255), ToTensorV2()]),
                ann_transform=None,
            )
            return HRC_WHUDataModule(
                root=self.root,
                train_pipeline=train_pipeline,
                val_pipeline=val_pipeline,
                test_pipeline=test_pipeline,
                batch_size=1,
            )
        elif experiment_name == "l8_biome_crop":
            train_pipeline = val_pipeline = test_pipeline = dict(
                all_transform=None,
                img_transform=albu.Compose([ToTensorV2()]),
                ann_transform=None,
            )
            return L8BiomeCropDataModule(
                root=self.root,
                train_pipeline=train_pipeline,
                val_pipeline=val_pipeline,
                test_pipeline=test_pipeline,
                batch_size=1,
            )
        raise ValueError(f"Experiment name {experiment_name} is not recognized.")

    @torch.no_grad()
    def inference(self, image: torch.Tensor) -> torch.tensor:
        logits:torch.Tensor = self.model(image)
        if isinstance(logits,tuple):
            logits = logits[0]
        preds = logits.argmax(dim=1).detach()
        return preds

    def run(self):

        metric = IoUMetric(
            num_classes=self.num_classes,
            iou_metrics=["mIoU", "mDice", "mFscore"],
            model_name=f"{self.dataset_name}_{self.model_name}",
        )

        for data in track(
            self.val_dataloader,
            description="evaling...",
            total=len(self.val_dataloader),
        ):
            img: torch.Tensor = data["img"].to(self.device)
            ann: torch.Tensor = data["ann"].to(self.device)

            pred = self.inference(img)
            metric.results.append(
                metric.intersect_and_union(pred, ann, self.num_classes, ignore_index=255)
            )

        result = metric.compute_metrics(metric.results)
        with open(f"{self.dataset_name}_{self.model_name}.json", "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

    def run_each(self):
        all_miou = []

        for data in track(
            self.val_dataloader,
            description="evaling...",
            total=len(self.val_dataloader),
        ):
            metric = IoUMetric(
            num_classes=self.num_classes,
            iou_metrics=["mIoU", "mDice", "mFscore"],
            model_name=f"{self.dataset_name}_{self.model_name}",
            )

            img: torch.Tensor = data["img"].to(self.device)
            ann: torch.Tensor = data["ann"].to(self.device)

            pred = self.inference(img)
            metric.results.append(
                metric.intersect_and_union(pred, ann, self.num_classes, ignore_index=255)
            )

            result = metric.compute_metrics(metric.results)
            all_miou.append(result['mIoU'])

        with open("/data/weix/SCTNet/kappamask_L8B_mIoUs.pkl", "wb") as f:
            pickle.dump(all_miou, f)


    def run_time(self):
        time_list = []
        N = len(self.val_dataloader)
        for data in tqdm.tqdm(self.val_dataloader):
            img: torch.Tensor = data["img"].to(self.device)
            time1 = time.time()
            pred = self.inference(img)
            time2 = time.time()
            time_list.append(time2-time1)
        time_list = np.asarray(time_list)
        print(N/np.sum(time_list))


    def throughputamp(self):
        for data in self.val_dataloader:
            images = data["img"].to(self.device)
            batch_size = images.shape[0]
            for i in range(50):
                with torch.cuda.amp.autocast():
                    self.inference(images)
            torch.cuda.synchronize()
            print(f"throughput averaged with 30 times")
            torch.cuda.reset_peak_memory_stats()
            tic1 = time.time()
            for i in range(30):
                with torch.cuda.amp.autocast():
                    self.inference(images)
            torch.cuda.synchronize()
            tic2 = time.time()
            print(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
            print(f"batch_size {batch_size} mem cost {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")
            return

    def export_onnx(self):
        for data in self.val_dataloader:
            images = data["img"].to(self.device)
            dummy_input = images
            torch.onnx.export(self.model, dummy_input, "sctnet.onnx", opset_version=12,
                          input_names=["input"], output_names=["output"],
                          dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})
            break


if __name__ == "__main__":
    dataset_name, model_name,model_path,gpu = get_args()
    Eval(dataset_name, model_name,model_path,gpu).run()
    # Eval(dataset_name, model_name,model_path,gpu).throughputamp()
    # Eval(dataset_name, model_name, model_path, gpu).export_onnx()
    # Eval(dataset_name, model_name, model_path, gpu).run_each()