import argparse

import torch.backends.cudnn as cudnn

from models.experimental import *
from utils.datasets import *
from utils.utils import *
from collections import defaultdict


def detect(out, source, pretrained_weights, custom_weights, view_img, imgsz, device, 
    conf_thres, iou_thres, classes, agnostic_nms, augment, supermarket_map, correct_class_name, save_img=False):
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load models
    model = attempt_load(pretrained_weights, map_location=device)  # load FP32 model
    custom_model = attempt_load(custom_weights, map_location=device)
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16
        custom_model.half()
    all_models = [model, custom_model]

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    custom_names = custom_model.module.names if hasattr(custom_model, 'module') else custom_model.names
    all_names = [names, custom_names]
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    colors = {'correct': [0, 255, 0], 'wrong': [0, 0, 255]}

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        all_bboxes = defaultdict(lambda: [])

        for m in range(len(all_models)):
            # Inference
            t1 = torch_utils.time_synchronized()
            pred = all_models[m](img, augment=augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
            t2 = torch_utils.time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                save_path = str(Path(out) / Path(p).name)
                # txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                # bboxes = defaultdict(lambda: [])

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, all_names[m][int(c)])  # add to string

                    # Write results
                    for *xyxy, conf, cls in det:
                        # if save_txt:  # Write to file
                        #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        #     with open(txt_path + '.txt', 'a') as f:
                        #         f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                        if all_names[m][int(cls)] in supermarket_map.values():
                            if save_img or view_img:  # Add bbox to image
                                label = '%s %.2f' % (all_names[m][int(cls)], conf)
                                temp = []
                                for tensor in xyxy:
                                    temp.append(tensor.item())
                                all_bboxes[all_names[m][int(cls)]].append(temp)
                                
                                if all_names[m][int(cls)] == correct_class_name:
                                # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                                    plot_one_box(xyxy, im0, label=label, color=colors['correct'], line_thickness=2)
                                else:
                                    plot_one_box(xyxy, im0, label=label, color=colors['wrong'], line_thickness=2)

                # Print time (inference + NMS)
                # print('%sDone. (%.3fs)' % (s, t2 - t1))

                # # Stream results
                # if view_img:
                #     cv2.imshow(p, im0)
                #     if cv2.waitKey(1) == ord('q'):  # q to quit
                #         raise StopIteration

                # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    print('Done. (%.3fs)' % (time.time() - t0))

    return all_bboxes, im0.shape


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
#     parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
#     parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
#     parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
#     parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
#     parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
#     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--view-img', action='store_true', help='display results')
#     parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
#     parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
#     parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
#     parser.add_argument('--augment', action='store_true', help='augmented inference')
#     parser.add_argument('--update', action='store_true', help='update all models')
#     opt = parser.parse_args()
#     print(opt)

#     with torch.no_grad():
#         if opt.update:  # update all models (to fix SourceChangeWarning)
#             for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3-spp.pt']:
#                 detect(opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.device, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, opt.augment)
#                 create_pretrained(opt.weights, opt.weights)
#         else:
#             detect(opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.device, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, opt.augment)
