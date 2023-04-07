// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Modified by Q-engineering 2023/04/04
//

#include <chrono>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "paddle_api.h" // NOLINT
#include "paddle_place.h"
#include "paddle_use_kernels.h"
#include "paddle_use_ops.h"

#include "cls_process.h"
#include "crnn_process.h"
#include "db_post_process.h"

using namespace paddle::lite_api; // NOLINT
using namespace std;

// fill tensor with mean and scale and trans layout: nhwc -> nchw, neon speed up
void NeonMeanScale(const float *din, float *dout, int size,
                   const std::vector<float> mean,
                   const std::vector<float> scale)
{
    if (mean.size() != 3 || scale.size() != 3) {
        std::cerr << "[ERROR] mean or scale size must equal to 3" << std::endl;
        exit(1);
    }
    float32x4_t vmean0 = vdupq_n_f32(mean[0]);
    float32x4_t vmean1 = vdupq_n_f32(mean[1]);
    float32x4_t vmean2 = vdupq_n_f32(mean[2]);
    float32x4_t vscale0 = vdupq_n_f32(scale[0]);
    float32x4_t vscale1 = vdupq_n_f32(scale[1]);
    float32x4_t vscale2 = vdupq_n_f32(scale[2]);

    float *dout_c0 = dout;
    float *dout_c1 = dout + size;
    float *dout_c2 = dout + size * 2;

    int i = 0;
    for (; i < size - 3; i += 4) {
        float32x4x3_t vin3 = vld3q_f32(din);
        float32x4_t vsub0 = vsubq_f32(vin3.val[0], vmean0);
        float32x4_t vsub1 = vsubq_f32(vin3.val[1], vmean1);
        float32x4_t vsub2 = vsubq_f32(vin3.val[2], vmean2);
        float32x4_t vs0 = vmulq_f32(vsub0, vscale0);
        float32x4_t vs1 = vmulq_f32(vsub1, vscale1);
        float32x4_t vs2 = vmulq_f32(vsub2, vscale2);
        vst1q_f32(dout_c0, vs0);
        vst1q_f32(dout_c1, vs1);
        vst1q_f32(dout_c2, vs2);

        din += 12;
        dout_c0 += 4;
        dout_c1 += 4;
        dout_c2 += 4;
    }
    for (; i < size; i++) {
        *(dout_c0++) = (*(din++) - mean[0]) * scale[0];
        *(dout_c1++) = (*(din++) - mean[1]) * scale[1];
        *(dout_c2++) = (*(din++) - mean[2]) * scale[2];
    }
}

// resize image to a size multiple of 32 which is required by the network
cv::Mat DetResizeImg(const cv::Mat img, int max_size_len,std::vector<float> &ratio_hw)
{
    int w = img.cols;
    int h = img.rows;

    float ratio = 1.f;
    int max_wh = w >= h ? w : h;
    if (max_wh > max_size_len) {
        if (h > w) ratio = static_cast<float>(max_size_len) / static_cast<float>(h);
        else       ratio = static_cast<float>(max_size_len) / static_cast<float>(w);
    }

    int resize_h = static_cast<int>(float(h) * ratio);
    int resize_w = static_cast<int>(float(w) * ratio);
    if (resize_h % 32 == 0)
        resize_h = resize_h;
    else if (resize_h / 32 < 1 + 1e-5)
        resize_h = 32;
    else
        resize_h = (resize_h / 32 - 1) * 32;

    if (resize_w % 32 == 0)
        resize_w = resize_w;
    else if (resize_w / 32 < 1 + 1e-5)
        resize_w = 32;
    else
        resize_w = (resize_w / 32 - 1) * 32;

    cv::Mat resize_img;
    cv::resize(img, resize_img, cv::Size(resize_w, resize_h));

    ratio_hw.push_back(static_cast<float>(resize_h) / static_cast<float>(h));
    ratio_hw.push_back(static_cast<float>(resize_w) / static_cast<float>(w));

    return resize_img;
}

void RunRecModel(std::vector<std::vector<std::vector<int>>> boxes, cv::Mat img,
                 std::shared_ptr<PaddlePredictor> predictor_crnn,
                 std::vector<std::string> &rec_text,
                 std::vector<float> &rec_text_score,
                 std::vector<std::string> charactor_dict,
                 int rec_image_height) {
    std::vector<float> mean = {0.5f, 0.5f, 0.5f};
    std::vector<float> scale = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};

    cv::Mat srcimg;
    img.copyTo(srcimg);
    cv::Mat crop_img;
    cv::Mat resize_img;

    for (size_t i=0; i<boxes.size(); i++) {
//    for (int i = boxes.size() - 1; i >= 0; i--) {
        crop_img = GetRotateCropImage(srcimg, boxes[i]);

        float wh_ratio = static_cast<float>(crop_img.cols) / static_cast<float>(crop_img.rows);

        resize_img = CrnnResizeImg(crop_img, wh_ratio, rec_image_height);
        resize_img.convertTo(resize_img, CV_32FC3, 1 / 255.f);

        const float *dimg = reinterpret_cast<const float *>(resize_img.data);

        std::unique_ptr<Tensor> input_tensor0(std::move(predictor_crnn->GetInput(0)));
        input_tensor0->Resize({1, 3, resize_img.rows, resize_img.cols});
        auto *data0 = input_tensor0->mutable_data<float>();

        NeonMeanScale(dimg, data0, resize_img.rows * resize_img.cols, mean, scale);
        //// Run CRNN predictor
        predictor_crnn->Run();

        // Get output and run postprocess
        std::unique_ptr<const Tensor> output_tensor0(std::move(predictor_crnn->GetOutput(0)));
        auto *predict_batch = output_tensor0->data<float>();
        auto predict_shape = output_tensor0->shape();

        // ctc decode
        std::string str_res;
        int argmax_idx;
        int last_index = 0;
        float score = 0.f;
        int count = 0;
        float max_value = 0.0f;

        for (int n = 0; n < predict_shape[1]; n++) {
            argmax_idx = int(Argmax(&predict_batch[n * predict_shape[2]], &predict_batch[(n + 1) * predict_shape[2]]));
            max_value = float(*std::max_element(&predict_batch[n * predict_shape[2]], &predict_batch[(n + 1) * predict_shape[2]]));
            if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
                score += max_value;
                count += 1;
                str_res += charactor_dict[argmax_idx];
            }
            last_index = argmax_idx;
        }
        score /= count;
        rec_text.push_back(str_res);
        rec_text_score.push_back(score);
    }
}

std::vector<std::vector<std::vector<int>>> RunDetModel(std::shared_ptr<PaddlePredictor> predictor, cv::Mat img,std::map<std::string, double> Config)
{
    // Read img
    int max_side_len = int(Config["max_side_len"]);
    int det_db_use_dilate = int(Config["det_db_use_dilate"]);

    cv::Mat srcimg;
    img.copyTo(srcimg);

    std::vector<float> ratio_hw;
    img = DetResizeImg(img, max_side_len, ratio_hw);
    cv::Mat img_fp;
    img.convertTo(img_fp, CV_32FC3, 1.0 / 255.f);

    // Prepare input data from image
    std::unique_ptr<Tensor> input_tensor0(std::move(predictor->GetInput(0)));
    input_tensor0->Resize({1, 3, img_fp.rows, img_fp.cols});
    auto *data0 = input_tensor0->mutable_data<float>();

    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> scale = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f};
    const float *dimg = reinterpret_cast<const float *>(img_fp.data);
    NeonMeanScale(dimg, data0, img_fp.rows * img_fp.cols, mean, scale);

    // Run predictor
    predictor->Run();

    // Get output and post process
    std::unique_ptr<const Tensor> output_tensor(std::move(predictor->GetOutput(0)));
    auto *outptr = output_tensor->data<float>();
    auto shape_out = output_tensor->shape();

    // Save output
    float pred[shape_out[2] * shape_out[3]];
    unsigned char cbuf[shape_out[2] * shape_out[3]];

    for (int i = 0; i < int(shape_out[2] * shape_out[3]); i++) {
        pred[i] = static_cast<float>(outptr[i]);
        cbuf[i] = static_cast<unsigned char>((outptr[i]) * 255);
    }

    cv::Mat cbuf_map(shape_out[2], shape_out[3], CV_8UC1,reinterpret_cast<unsigned char *>(cbuf));
    cv::Mat pred_map(shape_out[2], shape_out[3], CV_32F,reinterpret_cast<float *>(pred));

    const double threshold = double(Config["det_db_thresh"]) * 255;
    const double max_value = 255;
    cv::Mat bit_map;
    cv::threshold(cbuf_map, bit_map, threshold, max_value, cv::THRESH_BINARY);
    if (det_db_use_dilate == 1) {
        cv::Mat dilation_map;
        cv::Mat dila_ele = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
        cv::dilate(bit_map, dilation_map, dila_ele);
        bit_map = dilation_map;
    }
    auto boxes = BoxesFromBitmap(pred_map, bit_map, Config);

    std::vector<std::vector<std::vector<int>>> filter_boxes =  FilterTagDetRes(boxes, ratio_hw[0], ratio_hw[1], srcimg);

    return filter_boxes;
}

std::shared_ptr<PaddlePredictor> loadModel(std::string model_file, int num_threads)
{
    MobileConfig config;
    config.set_model_from_file(model_file);

    config.set_threads(num_threads);
    std::shared_ptr<PaddlePredictor> predictor = CreatePaddlePredictor<MobileConfig>(config);

    return predictor;
}

void Visualization(cv::Mat &frame, std::vector<std::vector<std::vector<int>>> boxes, std::vector<std::string> rec_text)
{
    int Wd;
    cv::Point rook_points[boxes.size()][4];

    for(size_t n = 0; n < boxes.size(); n++){
        for(size_t m = 0; m < boxes[0].size(); m++){
            rook_points[n][m] = cv::Point(static_cast<int>(boxes[n][m][0]), static_cast<int>(boxes[n][m][1]));
        }
    }
    for (size_t n = 0; n < boxes.size(); n++) {
        Wd = rook_points[n][1].x-rook_points[n][0].x;
        if(n < rec_text.size() && Wd>=30) {     //only large license plates
            const cv::Point *ppt[1] = {rook_points[n]};
            int npt[] = {4};
            int baseLine = 0;

            cv::polylines(frame, ppt, npt, 1, 1, CV_RGB(0, 255, 0), 2, 8, 0);

            cv::Size label_size = cv::getTextSize(rec_text[n], cv::FONT_HERSHEY_SIMPLEX, 0.7, 1, &baseLine);

            int x = rook_points[n][0].x;
            int y = rook_points[n][0].y - label_size.height - baseLine;
            if (y < 0) y = 0;
            if (x + label_size.width > frame.cols) x = frame.cols - label_size.width;

            cv::rectangle(frame, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cv::Scalar(255, 255, 255), -1);

            cv::putText(frame, rec_text[n], cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0));
        }

    }
}

std::vector<std::string> split(const std::string &str,const std::string &delim)
{
    std::vector<std::string> res;
    if ("" == str) return res;

    char *strs = new char[str.length() + 1];
    std::strcpy(strs, str.c_str());

    char *d = new char[delim.length() + 1];
    std::strcpy(d, delim.c_str());

    char *p = std::strtok(strs, d);
    while (p) {
        string s = p;
        res.push_back(s);
        p = std::strtok(NULL, d);
    }

    return res;
}

std::map<std::string, double> LoadConfigTxt(std::string config_path)
{
    auto config = ReadDict(config_path);

    std::map<std::string, double> dict;
    for(size_t i = 0; i < config.size(); i++) {
        std::vector<std::string> res = split(config[i], " ");
        dict[res[0]] = stod(res[1]);
    }
    return dict;
}

int main(int argc, char **argv) {

    float f;
    float FPS[16];
    int i, Fcnt=0;
    cv::Mat frame;
    chrono::steady_clock::time_point Tbegin, Tend;

    for(i=0;i<16;i++) FPS[i]=0.0;

    //// load config from txt file
    auto Config = LoadConfigTxt("./models/config.txt");
    int rec_image_height = int(Config["rec_image_height"]);
    auto charactor_dict = ReadDict("./models/ppocr_keys_v1.txt");
    charactor_dict.insert(charactor_dict.begin(), "#"); // blank char for ctc
    charactor_dict.push_back(" ");

    auto det_predictor = loadModel("./models/ch_PP-OCRv3_det_slim_opt.nb",4);
    auto rec_predictor = loadModel("./models/ch_PP-OCRv3_rec_slim_opt.nb",4);

    cv::VideoCapture cap("Highway.mp4");
    if (!cap.isOpened()) {
        cerr << "ERROR: Unable to open the camera" << endl;
        return 0;
    }

    cout << "Start grabbing, press ESC on Live window to terminate" << endl;
	while(1){
        cap >> frame;
        if (frame.empty()) {
            cerr << "ERROR: Unable to grab from the camera" << endl;
            break;
        }

        Tbegin = chrono::steady_clock::now();

        auto boxes = RunDetModel(det_predictor, frame, Config);

        std::vector<std::string> rec_text;
        std::vector<float> rec_text_score;

        RunRecModel(boxes, frame, rec_predictor, rec_text, rec_text_score, charactor_dict, rec_image_height);

        Visualization(frame, boxes, rec_text);

        Tend = chrono::steady_clock::now();
        //calculate frame rate
        f = chrono::duration_cast <chrono::milliseconds> (Tend - Tbegin).count();
        if(f>0.0) FPS[((Fcnt++)&0x0F)]=1000.0/f;
        for(f=0.0, i=0;i<16;i++){ f+=FPS[i]; }
        cv::putText(frame, cv::format("FPS %0.2f", f/16),cv::Point(10,20),cv::FONT_HERSHEY_SIMPLEX,0.6, cv::Scalar(0, 0, 255));

        //show output
        cv::imshow("RPi 4 - 1,9 GHz - 2 Mb RAM", frame);

        char esc = cv::waitKey(2);
        if(esc == 27) break;
    }

    cout << "Closing the camera" << endl;
    cv::destroyAllWindows();
    cout << "Bye!" << endl;

    return 0;
}
