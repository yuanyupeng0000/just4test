#include <vector>
#include "caffe/layers/absolute_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void AbsoluteLossLayer<Dtype>::Reshape(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
        LossLayer<Dtype>::Reshape(bottom, top); //define in LossLayer
        CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1)) //to make sure that input dimesion same
                << "Inputs must have the same dimension.";
        dis_.ReshapeLike(*bottom[0]);
        }

template <typename Dtype>
void AbsoluteLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top)
    {
        int count = bottom[0]->count(); //the total number of featuremap is 'count'
        caffe_sub(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(),
                dis_.mutable_cpu_data()); //diff_ = bottom[0] - bottom[1]
        Dtype loss_param = this->layer_param_.absolute_loss_param().dis(); //what's the relation between dis_ and dis ?
        Dtype abs_sum = caffe_cpu_abs_sum(count, dis_.cpu_data());
        Dtype loss = loss_param * abs_sum / bottom[0]->num();
        top[0]->mutable_cpu_data()[0] = loss;
    }

template <typename Dtype>
void AbsoluteLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
                                            const vector<Blob<Dtype>*>& bottom)
    {
        for(int i=0; i<2; ++i)
        {
            if(propagate_down[i])
            {
                //for input label, bottom propagate_down is zero
                const Dtype sign = (i == 0) ? 1 : -1;
                const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
                caffe_cpu_axpby(
                            bottom[i]->count(),
                            alpha,
                            dis_.cpu_data(),
                            Dtype(0),
                            bottom[i]->mutable_cpu_diff());
            }
        }
    }
#ifd CPU_ONLY
STUB_GPU(AbsoluteLossLayer);
#endif

INSTANTIATE_CLASS(AbsoluteLossLayer);
REGISTER_LAYER_CLASS(AbsoluteLoss);

} //namespace caffe
