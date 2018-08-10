#ifndef ABSOLUTE_LOSS_LAYER_H
#define ABSOLUTE_LOSS_LAYER_H

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"
namespace caffe {

template <typename Dtype>
class AbsoluteLossLayer : public LossLayer<Dtype>{
public:
    explicit AbsoluteLossLayer(const LayerParameter& param)
        : LossLayer<Dtype>(param), dis_() {}
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);
    virtual inline const char* type() const {return "AbsoluteLoss";}
    virtual inline bool AllowForceBackward(const int bottom_index) const {
        return true;
      }


    protected:
    /// @copydoc AbsoluteLossLayer
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);


    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


    Blob<Dtype> dis_;
};

} //namespace caffe
#endif // ABSOLUTE_LOSS_LAYER_H
