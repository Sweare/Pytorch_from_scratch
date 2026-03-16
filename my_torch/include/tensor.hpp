#pragma once

#include <cmath>
#include <concurrencysal.h>
#include <cstdio>
#include <vector>
#include <memory>
#include <numeric>
#include <type_traits>
#include <exception>
#include <algorithm>
#include <span>
#include <cassert>
#include <string>
#include <stdexcept>
#include <limits>
#include <thread>

// ============================================================
//  Tensor<T>  –  Core data structure
//
//  Responsibilities:
//    - Owns (via shared_ptr) or views a flat memory buffer
//    - Tracks shape, strides, and offset for N-D indexing
//    - Provides element access, slicing, view/reshape/transpose
//    - Basic element-wise arithmetic operators (+, -, +=, -=)
//
//  Math-heavy operations (matmul, sum, mean, max, relu …)
//  are exposed as free functions in ops.hpp so this file
//  stays focused on memory and indexing.
// ============================================================

template<typename T>
class Tensor {
// ---- internal helpers ----------------------------------------
    void computeStrides() {
        int sz = shape.size();
        if (sz == 0) return;
        strides.resize(sz);
        strides[sz - 1] = 1;
        // NOTE: use signed int, not size_t, to avoid infinite reverse loop
        for (int i = sz - 2; i >= 0; i--)
            strides[i] = strides[i + 1] * shape[i + 1];
    }

public:
    // ---- data members (public for now, like PyTorch's .data) ----
    std::vector<int64_t>               shape;
    std::vector<int64_t>               strides;
    std::shared_ptr<std::vector<T>>    data;
    int64_t                            offset = 0;

    std::vector<T>&       getData()       { return *data; }
    const std::vector<T>& getData() const { return *data; }

    // ==============================================================
    //  Constructors
    // ==============================================================

    // Allocating constructor: owns fresh memory
    explicit Tensor(std::vector<int64_t> s)
        : shape(s), data(std::make_shared<std::vector<T>>())
    {
        if (shape.empty())
            throw std::invalid_argument("0-dimension tensors do not exist.");
        if (shape.size() == 1)
            throw std::invalid_argument(
                "No 1-D tensors. Use {1,N} for a row or {N,1} for a column.");
        computeStrides();
        size_t total = 1;
        for (int dim : shape) total *= dim;
        data->resize(total);
    }

    // View constructor: shares existing memory, no allocation
    Tensor(std::vector<int64_t> s,
           std::shared_ptr<std::vector<T>> shared_data,
           int64_t new_offset)
        : shape(s), data(shared_data), offset(new_offset)
    {
        computeStrides();
    }

    // ==============================================================
    //  Shape / stride utilities
    // ==============================================================

    int64_t getSize() const {
        int64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }

    int64_t totalElements(std::span<const int64_t> s) const {
        int64_t n = 1;
        for (auto d : s) n *= d;
        return n;
    }
    int64_t totalElements(const std::vector<int64_t>& s) const {
        int64_t n = 1;
        for (auto d : s) n *= d;
        return n;
    }

    int64_t countBatches() const {
        int64_t n = 1;
        for (size_t i = 0; i < shape.size() - 2; i++) n *= shape[i];
        return n;
    }

    bool checkShape(const Tensor<T>& other) const {
        if (shape.size() != other.shape.size())
            return false;
        for (size_t i = 0; i < shape.size(); i++)
            if (shape[i] != other.shape[i]) return false;
        return true;
    }

    // Compute contiguous strides for an arbitrary shape (helper used by
    // view, reshape, isContiguous)
    std::vector<int64_t> computeStrides(const std::vector<int64_t>& tgt) const {
        int sz = tgt.size();
        std::vector<int64_t> st(sz, 0);
        if (sz == 0) return st;
        st[sz - 1] = 1;
        for (int i = sz - 2; i >= 0; i--)
            st[i] = st[i + 1] * tgt[i + 1];
        return st;
    }

    bool isContiguous() const {
        return computeStrides(shape) == strides;
    }

    // Physical copy of the data into a fresh, contiguous buffer
    Tensor makeContiguous() const {
        Tensor res(shape);
        std::vector<int64_t> id(shape.size(), 0);
        for (size_t i = 0; i < (size_t)res.getSize(); i++) {
            (*res.data)[i] = this->getElement(id);
            for (int64_t j = (int64_t)shape.size() - 1; j >= 0; j--) {
                id[j]++;
                if (id[j] < shape[j]) break;
                id[j] = 0;
            }
        }
        return res;
    }

    // ==============================================================
    //  View / reshape / transpose  (no copies unless necessary)
    // ==============================================================

    Tensor view(std::span<const int64_t> newShape) const {
        assert(totalElements(newShape) == totalElements(shape));
        assert(isContiguous());
        Tensor result = *this;
        result.shape   = std::vector<int64_t>(newShape.begin(), newShape.end());
        result.strides = computeStrides(result.shape);
        return result;
    }

    Tensor reshape(std::span<const int64_t> newShape) const {
        assert(totalElements(newShape) == totalElements(shape));
        if (isContiguous()) return view(newShape);
        return makeContiguous().view(newShape);
    }

    Tensor transpose(int64_t id1 = 0, int64_t id2 = 1) const {
        assert(id1 < (int64_t)shape.size() && id2 < (int64_t)shape.size());
        Tensor t = *this;
        std::swap(t.shape[id1],   t.shape[id2]);
        std::swap(t.strides[id1], t.strides[id2]);
        return t;
    }

    // ==============================================================
    //  Element access
    // ==============================================================

    // Checked variadic access (user-facing)
    template<typename... Args>
    T& getElement(Args... indices) {
        static_assert(sizeof...(Args) > 0, "Must provide indices");
        static_assert((std::is_integral_v<Args> && ...), "Indices must be integers");
        if (sizeof...(Args) != shape.size())
            throw std::invalid_argument(
                "Rank mismatch: expected " + std::to_string(shape.size()) +
                " indices, got " + std::to_string(sizeof...(Args)));
        int idx[] = { static_cast<int>(indices)... };
        int flat = 0;
        for (size_t i = 0; i < sizeof...(indices); i++) flat += idx[i] * strides[i];
        return (*data)[flat];
    }

    // Vector-index access (internal loops)
    T& getElement(const std::vector<int64_t>& idx) {
        if (idx.size() != shape.size()) throw std::invalid_argument("Rank mismatch");
        int64_t flat = 0;
        for (size_t i = 0; i < idx.size(); i++) flat += idx[i] * strides[i];
        return (*data)[flat];
    }
    const T& getElement(const std::vector<int64_t>& idx) const {
        if (idx.size() != shape.size()) throw std::invalid_argument("Rank mismatch");
        int64_t flat = 0;
        for (size_t i = 0; i < idx.size(); i++) flat += idx[i] * strides[i];
        return (*data)[flat];
    }

    // Fast variadic access via operator() — offset-aware so Views work
    template<typename... Args>
    T& operator()(Args... indices) noexcept {
        size_t i = 0;
        size_t flat = this->offset;
        ((flat += static_cast<size_t>(indices) * strides[i++]), ...);
        return (*data)[flat];
    }
    template<typename... Args>
    const T& operator()(Args... indices) const {
        size_t i = 0;
        size_t flat = this->offset;
        ((flat += static_cast<size_t>(indices) * strides[i++]), ...);
        return (*data)[flat];
    }

    // operator[] drops the outermost dimension and returns a view
    Tensor operator[](size_t index) const {
        if (shape.empty())    throw std::out_of_range("Cannot index a 0-D scalar.");
        if (index >= (size_t)shape[0]) throw std::out_of_range("Index out of bounds.");
        std::vector<int64_t> newShape(shape.begin() + 1, shape.end());
        int64_t new_offset = this->offset + (int64_t)index * this->strides[0];
        return Tensor(newShape, this->data, new_offset);
    }

    // ==============================================================
    //  Batch / row slicing helpers (used by matmul internally)
    // ==============================================================

    // Returns a 2-D view of the i-th matrix in a batched tensor
    Tensor<T> getMatrix(size_t batch_index) const {
        assert(shape.size() >= 2);
        int64_t rows = shape[shape.size() - 2];
        int64_t cols = shape[shape.size() - 1];
        Tensor<T> mat({rows, cols});
        mat.data   = this->data;
        mat.offset = batch_index * rows * cols;
        return mat;
    }

    // Broadcast-aware variant: maps a result-tensor batch index back to
    // the correct offset in a padded (possibly broadcast) input tensor
    Tensor<T> getMatrix(size_t batch_index,
                        const std::vector<int64_t>& res_shape,
                        const std::vector<int64_t>& padded_shape) const {
        assert(shape.size() >= 2);
        int64_t rows        = shape[shape.size() - 2];
        int64_t cols        = shape[shape.size() - 1];
        int64_t matrix_size = rows * cols;
        int64_t actual_offset = 0;
        int64_t stride        = matrix_size;
        int64_t remaining_i   = batch_index;

        for (int d = (int)res_shape.size() - 3; d >= 0; d--) {
            int64_t coord = remaining_i % res_shape[d];
            remaining_i /= res_shape[d];
            if (padded_shape[d] != 1) {
                actual_offset += coord * stride;
                stride *= padded_shape[d];
            }
        }
        Tensor<T> mat({rows, cols});
        mat.data   = this->data;
        mat.offset = actual_offset;
        return mat;
    }

    Tensor<T> getRows(int start, int end) const {
        int dimension = shape.size();
        int num_rows  = shape[dimension - 2];
        if (start < 0 || end < start || end > num_rows)
            throw std::out_of_range("Index out of bounds.");
        int nOffset = this->offset + (start * strides[dimension - 2]);
        return Tensor<T>({end - start, shape[dimension - 1]}, data, nOffset);
    }

    // ==============================================================
    //  Assignment
    // ==============================================================

    Tensor<T>& operator=(const Tensor<T>& other) {
        if (this == &other) return *this;
        if (!checkShape(other))
            throw std::invalid_argument("Shapes must match for assignment.");
        size_t total = 1;
        for (size_t d : shape) total *= d;
        T*       dst = this->data->data() + this->offset;
        const T* src = other.data->data() + other.offset;
        for (size_t i = 0; i < total; i++) dst[i] = src[i];
        return *this;
    }

    // ==============================================================
    //  Element-wise arithmetic
    // ==============================================================

    Tensor<T>& operator+=(const Tensor<T>& other) {
        // if (!checkShape(other))
        //     throw std::invalid_argument("Tensor sizes must match for addition.");
        if(!checkShape(other)){
            if(!checkBroadcastableElementwise(*this, other)){
                 throw std::invalid_argument("Tensor sizes must match for addition.");
            }

            if(this->shape.size()>=other.shape.size()){
                T* thisData=this->data->data()+this->offset;
                const T* otherData = other.data->data() + other.offset;
                for (size_t i = 0; i < this->getSize(); i++)
                    thisData[i] += otherData[i % other.getSize()];
                return  *this;

            }else{
                throw std::invalid_argument("Right side must be the smaller one for in-place broadcast.");
            }

            
        }
        T*       a = this->data->data() + this->offset;
        const T* b = other.data->data() + other.offset;
        for (size_t i = 0; i < this->getSize(); i++)
            a[i] += b[i];
        return *this;
    }

    // std::vector<int64_t>  makeBroadCast(const Tensor<T>& other){
    //     size_t diff=other.shape.size()-this->shape.size();
    //     std::vector<int64_t> new_shape;
    //     for(size_t i=0;i<diff;i++){
    //         new_shape.push_back(1);
    //     }
    //     for(size_t i =0; i<shape.size();i++){
    //         new_shape.push_back(shape[i]);
    //     }
    //     return  new_shape;

    // }
    Tensor<T> operator+(const Tensor<T>& other) const {
        if (this->getSize() >= other.getSize()) {
            Tensor<T> res = *this;
            res += other;
            return res;
        } else {
            Tensor<T> res = other;
            res += *this;
            return res;
        }
    }

    Tensor<T>& operator+=(const T value) {
        for (auto& v : *data) v += value;
        return *this;
    }
    Tensor<T> operator+(const T value) const {

        Tensor<T> res(shape);
        *res.data   = *data;
        res.strides = strides;
        res += value;
        return res;
    }
    bool checkBroadcastableElementwise(const Tensor<T>& A, const Tensor<T>& B) const {
        size_t i = A.shape.size(), j = B.shape.size();
        while (i > 0 && j > 0) {
            i--; j--;
            if (A.shape[i] != B.shape[j] && A.shape[i] != 1 && B.shape[j] != 1)
                return false;
        }
        return true;
    }
    Tensor<T>& operator-=(const Tensor<T>& other) {
        // if (!checkShape(other))
        //     throw std::invalid_argument("Tensor sizes must match for subtraction.");
                if(!checkShape(other)){
        if(!checkBroadcastableElementwise(*this, other)){
                 throw std::invalid_argument("Tensor sizes must match for addition.");
            }

            if(this->shape.size()>=other.shape.size()){
                T* thisData=this->data->data()+this->offset;
                const T* otherData = other.data->data() + other.offset;
                for (size_t i = 0; i < this->getSize(); i++)
                    thisData[i] -= otherData[i % other.getSize()];
                return  *this;

            }else{
                throw std::invalid_argument("Right side must be the smaller one for in-place broadcast.");
            }

            
        }
        T*       a = this->data->data() + this->offset;
        const T* b = other.data->data() + other.offset;
        for (size_t i = 0; i < data->size(); i++)
            a[i] -= b[i];
        return *this;
    }
    Tensor<T> operator-(const Tensor<T>& other) const {
        if (this->getSize() >= other.getSize()) {
                Tensor<T> res = *this;
                res -= other;
                return res;
            }
            throw std::invalid_argument("Right side must be smaller or equal for subtraction.");
    }
    static Tensor<T> zeros(std::vector<int64_t> shape){
        Tensor<T> t(shape);
        T* data =t.data->data();
        for ( size_t i=0;i<data->size();i++ ) {
            data[i]=0;
        }
        return t;
    }
    static Tensor<T> ones(std::vector<int64_t> shape){
        Tensor<T> t(shape);
        T* data =t.data->data();
        for ( size_t i=0;i<data->size();i++ ) {
            data[i]=1;
        }
        return t;
    }
    void fill(T val){
     
        T* ptr = this->data->data() + this->offset;
        for ( size_t i=0;i<data->size();i++ ) {
            ptr[i]=val;
        }
       
    }
    void fillCoor(std::vector<int64_t> coor,T val){
        if(coor.size()!=this->shape.size()) throw std::invalid_argument("fill: wrong number of coordinates");
        T* el=getElement(coor);
        *el = val;
        
        
    }
    Tensor<T> hadamard(const Tensor<T>& other ) const{
        if(this->checkShape(other)) throw std::invalid_argument("must have the same shape");
        T* thisData =data->data();
        T* otherData =other.data->data();
        Tensor<T> res(this->shape);
        T* resData=res.data->data();
        for(size_t i=0;i<getSize();i++){
            resData[i]=thisData[i]*otherData[i];
        }
        return res;

    }
    Tensor<T> exp() const{
        Tensor<T> res(this->shape);
        T* thisData=this->data->data();
        T* resData=res.data->data();
        for(size_t i=0;i<this->getSize();i++){
            resData[i]=exp(thisData[i]);//TODO make this big numebr safe
        }
        return res;
    }
    Tensor<T> log() const{
        Tensor<T> res(this->shape);
        T* thisData=this->data->data();
        T* resData=res.data->data();
        for(size_t i=0;i<this->getSize();i++){
            // 1e-7 is standard, small enough to not mess up math, big enough to prevent infinity.
            resData[i]=log(thisData[i]+ 1e-7);
        }
        return res;
    }
    Tensor<T> relu() const{
        Tensor<T> res(this->shape);
        T* thisData=this->data->data();
        T* resData=res.data->data();
        for(size_t i=0;i<this->getSize();i++){
            
            resData[i] = (thisData[i] > 0) ? thisData[i] : static_cast<T>(0);//because we can't just say to float be 0
        }
        return res;
    }
    Tensor<T> sigmoid() const{
        Tensor<T> res(this->shape);
        T* thisData=this->data->data();
        T* resData=res.data->data();
        for(size_t i=0;i<this->getSize();i++){
            T neg=-thisData[i];
            T div=static_cast<T>(1)+exp(neg);
            T val= static_cast<T>(1)/div;
            resData[i] = val;
        }
        return res;
    }
};

