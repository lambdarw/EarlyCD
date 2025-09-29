import os
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import warnings
import importlib
# from mcd.src.dataset import MCD_collate_fn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from usfutils.logger import get_root_logger
from usfutils.config import load_yaml, copy_opt_file
from usfutils.dir import scandir
from usfutils.utils import set_seed_everything
from usfutils.load import instantiate_from_config
from usfutils.format import dict_to_str
from mcd.utils.metrics import metrics, reg_metrics
from torch.utils.tensorboard import SummaryWriter
from mcd.module.main_model import MCDModel, \
    MCDModel_wo_taskMOE, MCDModel_wo_guideAttn
from mcd.utils.metrics import huber_loss
import random
import numpy as np

warnings.filterwarnings("ignore")
current_dir = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    dataNumber = 4
    if dataNumber == 0:
        is_cold_start = True  # 是否冷启动 默认F表示不是冷启动
    else:
        is_cold_start = False
    dataset_name = 'mmcd'  # mmcd, ours
    task = 'MCDModel'  # MCDModel, MCDModel_wo_guideAttn, MCDModel_wo_taskMOE,

    if dataset_name == 'mmcd':
        config_path = "mcd/config/train.yaml"
        module = importlib.import_module("mcd.src.dataset")
    else:
        config_path = "mcd/config/ours_train.yaml"
        module = importlib.import_module("mcd.src.ours_dataset")

    MCD_collate_fn = getattr(module, "MCD_collate_fn")

    # 加载模型
    config = load_yaml(config_path)
    device = "cuda" if (torch.cuda.is_available() and config.gpu) else "cpu"
    if task == 'MCDModel':
        model = MCDModel(dataNumber, config.model, is_cold_start)
    elif task == 'MCDModel_wo_taskMOE':
        model = MCDModel_wo_taskMOE(dataNumber, config.model, is_cold_start)
    else:
        model = MCDModel_wo_guideAttn(dataNumber, config.model, is_cold_start)
    loss_list = []

    set_seed_everything(seed=config.seed)
    experiments_root = os.path.join(current_dir, "experiments/" + config.name)
    state_dict_path = os.path.join(
        experiments_root, f"model_dir/{config.baseline_type}_{config.log_type}"
    )
    if not os.path.exists(state_dict_path):
        os.makedirs(state_dict_path)
    copy_opt_file(config_path, experiments_root)
    logger = get_root_logger(
        log_path=experiments_root,
        logger_name=f"{config.baseline_type}_{config.log_type}"
    )
    logger.info(dict_to_str(config))

    dataset_train = instantiate_from_config(config.data.train)
    dataset_valid = instantiate_from_config(config.data.valid)
    dataset_test = instantiate_from_config(config.data.test)
    train_dataloader = DataLoader(
        dataset=dataset_train,
        batch_size=config.batch_size,
        num_workers=config.num_works,
        collate_fn=MCD_collate_fn,
        shuffle=True
    )
    valid_dataloader = DataLoader(
        dataset=dataset_valid,
        batch_size=config.batch_size,
        num_workers=config.num_works,
        collate_fn=MCD_collate_fn,
        shuffle=True
    )
    test_dataloader = DataLoader(
        dataset=dataset_test,
        batch_size=config.batch_size,
        num_workers=config.num_works,
        collate_fn=MCD_collate_fn,
        shuffle=True
    )
    model.to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer, max_lr=1e-2, steps_per_epoch=len(train_dataloader), epochs=config.epochs
    # )

    is_early_stop = False
    current_iter = 0
    best_test_acc = 0
    best_valid_acc = 0
    best_epoch = 0
    writer = SummaryWriter("logs")
    test_best_results = ""
    reg_test_best_results = ""
    for epoch in range(config.epochs):
        if is_early_stop:
            break
        t_predict = []
        t_label = []
        r_predict = []
        r_label = []
        # train
        for idx, data in enumerate(train_dataloader):
            current_iter += 1
            for k, v in data.items():
                data[k] = v.to(device)
            label = data.pop("label")
            hotNumLabel = data.pop("hotNums")
            if task == 'MCDModel_wo_taskMOE':
                cls_output, reg_output = model(**data, device=device)
                _, predicts = torch.max(cls_output, 1)
                loss = criterion(cls_output, label)
                reg_loss = huber_loss(reg_output, hotNumLabel)
                loss += config.alpha * reg_loss
                loss_list.append(loss.item())
            else:
                cls_output, reg_output, moe_loss_cls, moe_loss_reg = model(**data, device=device)
                _, predicts = torch.max(cls_output, 1)
                # loss = criterion(cls_output, label)
                # reg_loss = huber_loss(reg_output, hotNumLabel)
                loss = (criterion(cls_output, label) + moe_loss_cls)
                reg_loss = huber_loss(reg_output, hotNumLabel) + moe_loss_reg
                loss += config.alpha * reg_loss
                loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (current_iter + 1) % config.log_freq == 0:
                logger.info(
                    f"Epoch:{epoch}, Data: {idx}/{len(dataset_train) // config.batch_size}, Loss: {round(loss.item(), 5)}"
                )
            t_label.extend(label.detach().cpu().numpy().tolist())
            t_predict.extend(predicts.detach().cpu().numpy().tolist())
            r_label.extend(hotNumLabel.detach().cpu().numpy().tolist())
            r_predict.extend(reg_output.detach().cpu().numpy().tolist())


        results1 = metrics(t_label, t_predict)
        reg_results1 = reg_metrics(r_label, r_predict)
        logger.info(f"After epoch {epoch}, training results: {results1} {reg_results1}")
        writer.add_scalar(
            "train/loss", round(float(loss.detach().cpu()), 4), global_step=epoch
        )
        writer.add_scalar("train/accuracy", results1["f1"], global_step=epoch)
        # valid
        t_predict.clear()
        t_label.clear()
        r_predict.clear()
        r_label.clear()
        for idx, data in enumerate(valid_dataloader):
            for k, v in data.items():
                data[k] = v.to(device)
            label = data.pop("label")
            hotNumLabel = data.pop("hotNums")
            if task == 'MCDModel_wo_taskMOE':
                with torch.no_grad():
                    cls_output, reg_output = model(**data, device=device)
                    _, predicts = torch.max(cls_output, 1)
                    val_loss = criterion(cls_output, label)
                    reg_loss = huber_loss(reg_output, hotNumLabel)
                    val_loss += config.alpha * reg_loss
            else:
                with torch.no_grad():
                    cls_output, reg_output, moe_loss_cls, moe_loss_reg = model(**data, device=device)
                    _, predicts = torch.max(cls_output, 1)
                    # val_loss = criterion(cls_output, label)
                    # reg_loss = huber_loss(reg_output, hotNumLabel)
                    val_loss = (criterion(cls_output, label) + moe_loss_cls)
                    reg_loss = huber_loss(reg_output, hotNumLabel) + moe_loss_reg
                    val_loss += config.alpha * reg_loss
            t_label.extend(label.detach().cpu().numpy().tolist())
            t_predict.extend(predicts.detach().cpu().numpy().tolist())
            r_label.extend(hotNumLabel.detach().cpu().numpy().tolist())
            r_predict.extend(reg_output.detach().cpu().numpy().tolist())
        results2 = metrics(t_label, t_predict)
        reg_results2 = reg_metrics(r_label, r_predict)
        writer.add_scalar(
            "valid/loss", round(float(val_loss.detach().cpu()), 4), global_step=epoch
        )
        writer.add_scalar("valid/accuracy", results2["f1"], global_step=epoch)
        logger.info(f"After epoch {epoch}, validing results: {results2} {reg_results2}")
        if results2["f1"] > best_valid_acc:
            best_valid_acc = results2["f1"]
            best_epoch = epoch + 1
            if best_valid_acc > config.save_threshold:
                try:
                    for remove_file in scandir(
                        state_dict_path, suffix=".pth", full_path=True
                    ):
                        os.remove(remove_file)
                    save_path = os.path.join(
                        state_dict_path, f"{config.baseline_type}_epoch{best_epoch}_b.pth"
                    )
                    torch.save(model.state_dict(), save_path)
                    logger.info(f"Saved {save_path}")
                    #
                    state_dict = torch.load(save_path)
                    model.load_state_dict(state_dict)
                except:
                    pass
        else:
            if epoch - best_epoch >= config.epoch_stop - 1:
                is_early_stop = True
                logger.info(f"Early Stopping on Epoch {epoch}...")
                save_path = os.path.join(
                    state_dict_path, f"{config.baseline_type}_epoch{epoch}_l.pth"
                )
                torch.save(model.state_dict(), save_path)
                #
                state_dict = torch.load(save_path)
                model.load_state_dict(state_dict)
        # test
        t_predict.clear()
        t_label.clear()
        r_predict.clear()
        r_label.clear()
        for idx, data in enumerate(test_dataloader):
            for k, v in data.items():
                data[k] = v.to(device)
            label = data.pop("label")
            hotNumLabel = data.pop("hotNums")
            if task == 'MCDModel_wo_taskMOE':
                with torch.no_grad():
                    cls_output, reg_output = model(**data, device=device)
                    _, predicts = torch.max(cls_output, 1)
                    loss = (criterion(cls_output, label))
                    reg_loss = huber_loss(reg_output, hotNumLabel)
                    loss += config.alpha * reg_loss
            else:
                with torch.no_grad():
                    cls_output, reg_output, moe_loss_cls, moe_loss_reg = model(**data, device=device)  # , moe_loss_cls, moe_loss_reg
                    _, predicts = torch.max(cls_output, 1)
                    # loss = criterion(cls_output, label)
                    # reg_loss = huber_loss(reg_output, hotNumLabel)
                    loss = (criterion(cls_output, label) + moe_loss_cls)
                    reg_loss = huber_loss(reg_output, hotNumLabel) + moe_loss_reg
                    loss += config.alpha * reg_loss
            t_label.extend(label.detach().cpu().numpy().tolist())
            t_predict.extend(predicts.detach().cpu().numpy().tolist())
            r_label.extend(hotNumLabel.detach().cpu().numpy().tolist())
            r_predict.extend(reg_output.detach().cpu().numpy().tolist())
        results3 = metrics(t_label, t_predict)
        reg_results3 = reg_metrics(r_label, r_predict)
        if results3["f1"] > best_test_acc:
            test_best_results = results3
            reg_test_best_results = reg_results3
            best_test_acc = results3["f1"]
        writer.add_scalar(
            "test/loss", round(float(loss.detach().cpu()), 4), global_step=epoch
        )
        writer.add_scalar("test/accuracy", results3["f1"], global_step=epoch)
        logger.info(
            f"{config.baseline_type}: After epoch {epoch}, testing results: {results3} {reg_results3}"
        )
        logger.info(f"{config.baseline_type}: Testing best results: {test_best_results} {reg_test_best_results}")

    #     # scheduler.step()  # 调度器
    #     if epoch % 5 == 0:  # 10 个epoch绘画一个损失函数点，可以自定义
    #         print(f"epoch {epoch}: loss {loss.item()}")
    #         # 更新损失曲线
    #         plt.cla()
    #         plt.plot(loss_list)
    #         plt.xlabel("epoch")
    #         plt.ylabel("loss")
    #         plt.pause(0.01)
    # plt.show()
