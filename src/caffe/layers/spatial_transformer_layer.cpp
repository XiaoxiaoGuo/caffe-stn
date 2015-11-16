#include <cmath>
#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/custom_layers.hpp"

namespace caffe {

    template <typename Dtype>
    void SpatialTransformerLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top) {
    }

    template <typename Dtype>
    void SpatialTransformerLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        top[0]->ReshapeLike(*bottom[0]);
        CHECK_EQ(bottom[1]->shape(1), 6) << "Second blob should be 6-dimension theta";
        num_ = bottom[0]->shape()[0];
        channel_ = bottom[0]->shape()[1];
        height_ = bottom[0]->shape()[2];
        width_ = bottom[0]->shape()[3];
        map_size_ = width_ * height_;

        // init target coordinate
        vector<int> target_shape;
        target_shape.push_back(1);
        target_shape.push_back(3);
        target_shape.push_back(height_);
        target_shape.push_back(width_);
        target_.Reshape(target_shape);
        Dtype* target_data = target_.mutable_cpu_data();

        for (int h = 0; h < height_; ++h) {
            for (int w = 0; w < width_; ++w) {
                // for x;
                target_data[target_.offset(0, 0, h, w)] = (Dtype) w / (Dtype) (width_ - 1) * 2. - 1.;
                // for y
                target_data[target_.offset(0, 1, h, w)] = (Dtype) h / (Dtype) (height_ - 1) * 2. - 1.;
                // for constant
                target_data[target_.offset(0, 2, h, w)] = (Dtype) 1.0;
            }
        }

        // create source coordinates
        vector<int> source_shape;
        source_shape.push_back(num_);
        source_shape.push_back(2);
        source_shape.push_back(height_);
        source_shape.push_back(width_);
        source_.Reshape(source_shape);
        
        // create source range for bilinear sampling
        vector<int> source_range_shape;
        source_range_shape.push_back(num_);
        source_range_shape.push_back(height_);
        source_range_shape.push_back(width_);
        source_range_shape.push_back(4);
        source_range_.Reshape(source_range_shape);
        
        // create source gradient cache for different channels
        vector<int> source_grad_shape;
        source_grad_shape.push_back(channel_);
        source_grad_shape.push_back(num_);
        source_grad_shape.push_back(2);
        source_grad_shape.push_back(height_);
        source_grad_shape.push_back(width_);
        source_grad_cache_.Reshape(source_grad_shape);
        
        vector<int> all_ones_shape;
        all_ones_shape.push_back(channel_);
        source_grad_op_.Reshape(all_ones_shape);
        caffe_set<Dtype>(channel_, 1, source_grad_op_.mutable_cpu_data());
    }

    template <typename Dtype>
    void SpatialTransformerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top) {
        Dtype* top_data = top[0]->mutable_cpu_data();
        const Dtype* theta_data = bottom[1]->cpu_data();
        const Dtype* target_data = target_.cpu_data();

        Dtype* source_data = source_.mutable_cpu_data();
        int* source_range_data = source_range_.mutable_cpu_data();
        caffe_set<Dtype>(top[0]->count(), 0, top_data);
        for (int n = 0; n < num_; ++n) {
            // compute source coordinate 
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 2, map_size_, 3, Dtype(1.0),
                    theta_data + n * 6, target_data, Dtype(0.), source_data + n * 2 * map_size_);
            // compute source in real source coordinate range
            caffe_add_scalar(2 * map_size_, (Dtype)1. , source_data + n * 2 * map_size_);
            caffe_scal<Dtype>(map_size_, (Dtype) (width_ - 1) / (Dtype) 2., source_data + n * 2 * map_size_);
            caffe_scal<Dtype>(map_size_, (Dtype) (height_ - 1) / (Dtype) 2., source_data + n*2*map_size_+map_size_);
            
            
            // compute U given source coordinate: O(W*H)
            for (int h = 0; h < height_; ++h) {
                for (int w = 0; w < width_; ++w) {
                    Dtype x = source_data[source_.offset(n, 0, h, w)];
                    Dtype y = source_data[source_.offset(n, 1, h, w)];

                    //O(C)
                    int w_min = (floor(x) > 0) ? floor(x) : 0;
                    int w_max = (ceil(x) < width_ - 1) ? ceil(x) : (width_ - 1);
                    int h_min = (floor(y) > 0) ? floor(y) : 0;
                    int h_max = (ceil(y) < height_ - 1) ? ceil(y) : (height_ - 1);
                    source_range_data[source_range_.offset(n,h,w,0)] = w_min;
                    source_range_data[source_range_.offset(n,h,w,1)] = w_max;
                    source_range_data[source_range_.offset(n,h,w,2)] = h_min;
                    source_range_data[source_range_.offset(n,h,w,3)] = h_max;
                    for (int hh = h_min; hh <= h_max; ++hh) {
                        for (int ww = w_min; ww <= w_max; ++ww) {
                            for (int c = 0; c < channel_; ++c) {
                                top_data[top[0]->offset(n, c, h, w)] += bottom[0]->data_at(n, c, hh, ww)*(1 - fabs(x - ww)) * (1 - fabs(y - hh));
                            }
                        }
                    }
                }
            }
        }
    }

    template <typename Dtype>
    void SpatialTransformerLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down,
            const vector<Blob<Dtype>*>& bottom) {
        // @IMPRV current version ignores propagate_down signal
        const Dtype* top_diff = top[0]->cpu_diff();
        Dtype* data_diff = bottom[0]->mutable_cpu_diff();
        Dtype* theta_diff = bottom[1]->mutable_cpu_diff();
                const Dtype* target_data = target_.cpu_data();
        const Dtype* source_data = source_.cpu_data();
        const int* source_range_data = source_range_.cpu_data();
        Dtype* source_diff = source_.mutable_cpu_diff();

        caffe_set<Dtype>(bottom[0]->count(), 0, data_diff);
//        caffe_set<Dtype>(source_.count(), 0, source_diff);

        for (int n = 0; n < num_; ++n) {
            for (int h = 0; h < height_; ++h) {
                for (int w = 0; w < width_; ++w) {
                    Dtype x = source_data[source_.offset(n, 0, h, w)];
                    Dtype y = source_data[source_.offset(n, 1, h, w)];
                    int w_min = source_range_data[source_range_.offset(n,h,w,0)];
                    int w_max = source_range_data[source_range_.offset(n,h,w,1)];
                    int h_min = source_range_data[source_range_.offset(n,h,w,2)];
                    int h_max = source_range_data[source_range_.offset(n,h,w,3)];
                    Dtype tmp_source_x = 0;
                    Dtype tmp_source_y = 0;
                  
                    const Dtype width_const = (Dtype)(width_ - 1) / (Dtype)2.;
                    const Dtype height_const = (Dtype)(height_ - 1) / (Dtype)2.;
                    for (int hh = h_min; hh <= h_max; ++hh) {
                        for (int ww = w_min; ww <= w_max; ++ww) {
                            int sign_x = caffe_sign<Dtype>(ww - x);
                            int sign_y = caffe_sign<Dtype>(hh - y);//(y <= (Dtype)hh ) ? 1 : -1;

                            for (int c = 0; c < channel_; ++c) {
                                // d(L)/d(U^{c}_{nm})=\sum_{j} d(L)/d(V^{c}_{j}) * d(V^{c}_{j})/d(U^{c}_{nm})
                                // bottom_diff[(n,c,hh,ww)]=\sum_{j} top_diff[(n,c,h,w)] * eq(6) (an error)
                                Dtype buffer = top_diff[top[0]->offset(n, c, h, w)];
                                data_diff[bottom[0]->offset(n, c, hh, ww)] += buffer * (1 - fabs(x - ww)) * (1 - fabs(y - hh));
                                // d(L)/d(x_{j})=\sum_{c} d(L)/d(V^{c}_j)*d(V^{c}_j)/d(x_{j})
                                // source_diff[(n,0,h,w)] = \sum_{c} top[(n,c,h,w)] * \sum_{nm} U_{nm} max
                                buffer *= bottom[0]->data_at(n,c,hh,ww);
                                tmp_source_x += buffer*(1-fabs(y-hh))*sign_x * width_const;
                                tmp_source_y += buffer*(1-fabs(x-ww))*sign_y * height_const;
                                
                            }
                        }
                    }
                    source_diff[source_.offset(n,0,h,w)] = tmp_source_x; 
                    source_diff[source_.offset(n,1,h,w)] = tmp_source_y;
                }
            }
            // d(L)/d(theta)
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 2, 3, map_size_,
                    (Dtype)1., source_diff + n * 2 * map_size_, target_data, (Dtype)0., theta_diff + n * 6);
        }
    }


#ifdef CPU_ONLY
    STUB_GPU(SpatialTransformerLayer);
#endif

    INSTANTIATE_CLASS(SpatialTransformerLayer);

} // namespace caffe
