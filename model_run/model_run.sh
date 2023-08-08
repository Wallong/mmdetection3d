#!/bin/bash

flask --app=/mmdetection3d/model_run/model_test run --host=0.0.0.0 --port=5000

# never exit，此处是为了运行完上条应用服务后，有对应的前台进程
tail -f /dev/null