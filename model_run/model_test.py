import re
import logging

import requests
from flask import Flask, request, jsonify

import numpy as np
import pandas as pd

from timing import Timing
from mmdet3d.apis import init_model, inference_detector

# 使用的config和checkpoint路径
# PointPillars
# config_file='/mmdetection3d/ckpt/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py'
# checkpoint_file='/mmdetection3d/ckpt/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth'
# class_names = [
#     'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
#     'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
# ]

# CenterPoint
config_file='/mmdetection3d/ckpt/centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d.py'
checkpoint_file='/mmdetection3d/ckpt/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20220810_025930-657f67e0.pth'
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

app = Flask(__name__)


# 初始化模型
model = init_model(
    config_file, 
    checkpoint_file
)

def build_item_error(code: str, message: str, id: int = None):
    return {
        "id": id,
        "code": code,
        "message": message
    }

@app.route("/pointCloud/modelPredict", methods=['POST'])
def model_predict():
    # 从Xtreme1发送过来的请求中获取datas字段的值
    datas = request.json['datas']
    logging.info(datas)
    
    data_result = []
    for data in datas:
        if not isinstance(data, dict):
            return build_item_error("InvalidArgument", "data must be a dictionary")
        
        id = data.get("id", None)
        if id is None:
            return build_item_error("InvalidArgument", 'missing "id"')

        pcd_url = data.get("pointCloudUrl", None)
        if pcd_url is None:
            return build_item_error("InvalidArgument", 'missing "pointCloudUrl"')
  
        # 提前创建保存结果的字典
        single_result = {
            "id": id,
            "code": "OK",
            "message": "",
            "objects": []
        }

        try:
            t = Timing()
            logging.info(f"{'-'*10} {pcd_url} {'-'*10}")

            # 从url中下载pcd
            resp = requests.get(
                data["pointCloudUrl"], 
                allow_redirects=True
            )
            t.log_interval(f"DOWNLOAD pcd({len(resp.content)/1024/1024:.2g}MB)")
                    
            # 将pcd中的数据解析为数组
            pcd_arr = parse_array(resp.content)
            
            # 调用模型进行预测
            result, data = inference_detector(
                model=model,
                pcds=pcd_arr
            )
            
            # 将模型预测结果转换成Xtreme1接受的格式
            single_result["objects"] = parse_detection(result)
        except Exception as e:
            logging.exception(e)
        
        data_result.append(single_result)
    
    # 所有结果保存成一个字典, 作为响应
    result = {
        "code": "OK",
        "message": "",
        "data": data_result
    }
    
    return jsonify(result)

numpy_pcd_type_mappings = {
    ('F', 4): np.dtype('float32'),
    ('F', 8): np.dtype('float64'),
    ('U', 1): np.dtype('uint8'),
    ('U', 2): np.dtype('uint16'),
    ('U', 4): np.dtype('uint32'),
    ('U', 8): np.dtype('uint64'),
    ('I', 2): np.dtype('int16'),
    ('I', 4): np.dtype('int32'),
    ('I', 8): np.dtype('int64')
}

def min_max_scaler(
    data,
    value_range = (0, 1)
):
    d_min = np.min(data)    
    d_max = np.max(data) 
    v_min, v_max = value_range

    return v_min + (v_max - v_min) / (d_max - d_min) * (data - d_min)

def parse_array(
    raw_pcd,
    needed_cols=('x', 'y', 'z', 'i', 'intensity')
):
    # 使用正则提取pcd中的关键信息
    fields = re.search(rb'\nFIELDS (.*?)\n', raw_pcd).groups()[0].decode().split(' ')
    size = list(map(int, re.search(rb'\nSIZE (.*?)\n', raw_pcd).groups()[0].decode().split(' ')))
    field_type = re.search(rb'\nTYPE (.*?)\n', raw_pcd).groups()[0].decode().split(' ')
    data = re.search(rb'\nDATA .*?\n([\s\S]*)', raw_pcd).groups()[0]
    
    # 将bytes格式数据转化为np.array
    dtypes = [(fields[i], numpy_pcd_type_mappings[x]) for i, x in enumerate(zip(field_type, size))]
    total_arr = np.frombuffer(data, dtype=dtypes)
    
    # 只留下需要的列
    left_fields = [x for x in fields if x in needed_cols]
    pcd_df = pd.DataFrame(total_arr)[left_fields]
    pcd_df.rename(columns={'intensity': 'i'}, inplace=True)
    
    # 归一化（可选）
    pcd_df['i'] = min_max_scaler(pcd_df['i'].values)

    # 添加ring id
    pcd_df.insert(pcd_df.shape[1], 'r', 0)
    
    return pcd_df.values

def parse_detection(
    detection
):
    scores_list = detection.pred_instances_3d.scores_3d.tolist()
    labels_list = detection.pred_instances_3d.labels_3d.tolist()
    bboxes_list = detection.pred_instances_3d.bboxes_3d.tensor.tolist()
    
    return [
        {
            "label": class_names[labels_list[i]].upper(),
            "confidence": scores_list[i],
            "x": bboxes_list[i][0],
            "y": bboxes_list[i][1],
            "z": bboxes_list[i][2] + bboxes_list[i][5]/2,
            "dx": bboxes_list[i][3],
            "dy": bboxes_list[i][4],
            "dz": bboxes_list[i][5],
            "rotX": 0,
            "rotY": 0,
            "rotZ": bboxes_list[i][6],
        }
        for i in range(len(scores_list))
    ]

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)