#ifndef TENSOR_HPP
#define TENSOR_HPP

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
template<typename T>
class Tensor {
public:
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    std::shared_ptr<std::vector<T>> data;
    int64_t offset = 0;

    void computeStrides() {
        int size = shape.size();
        if (size == 0) return;

        strides.resize(size);
        strides[size - 1] = 1;

        // PRO TIP: Use 'int' or 'long' for reverse loops. 
        // 'size_t' is unsigned, so 'i >= 0' is ALWAYS true (infinite loop!)
        for (int i = size - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
    std::vector<T>& getData() { return *data; }
    const std::vector<T>& getData() const { return *data; }

//public:
    // Constructor
    Tensor(std::vector<int64_t> s) : shape(s), data(std::make_shared<std::vector<T>>()) {
    if (shape.empty()) {
        throw std::invalid_argument("0-dimension tensors do not exist.");
    }
    if (shape.size() == 1) {
        throw std::invalid_argument("There are no 1D tensors. If you want a vector, use {1, N} for a row or {N, 1} for a column.");
    }
        computeStrides();
        // Calculate total size and resize data
        size_t total_size = 1;
        for(int dim : shape) total_size *= dim;
        data->resize(total_size);
    }
    Tensor(std::vector<int64_t> s, std::shared_ptr<std::vector<T>> shared_data, int64_t new_offset) 
    : shape(s), data(shared_data), offset(new_offset) {
        // We just calculate the strides for the new shape. We DO NOT allocate memory!
        computeStrides(); 
    }
    std::vector<int64_t> computeStrides(const std::vector<int64_t>& target_shape) const {
        int size = target_shape.size();
        std::vector<int64_t> new_strides(size, 0);
        if (size == 0) return new_strides;

        new_strides[size - 1] = 1;
        for (int i = size - 2; i >= 0; i--) {
            new_strides[i] = new_strides[i + 1] * target_shape[i + 1];
        }
        return new_strides;
    }

    Tensor transpose(int64_t id1=0, int64_t id2=1) const{
        assert(id1 < shape.size() && id2 < shape.size());
        Tensor transp=*this;
        
        std::swap(transp.shape[id1],transp.shape[id2]);
        std::swap(transp.strides[id1],transp.strides[id2]);
        return transp;
    }
    Tensor view(std::span<const int64_t> newShape) const{
        // check total elements match
        assert(totalElements(newShape) == totalElements(shape));
        // check contiguous
        assert(isContiguous());
        
        Tensor result = *this;
        result.shape = std::vector<int64_t>(newShape.begin(), newShape.end());
        result.strides = computeStrides(std::vector<int64_t>(newShape.begin(), newShape.end()));  // fresh strides from new shape
        return result;
    }
    Tensor reshape(std::span<const int64_t> newShape) const{
        assert(totalElements(newShape) == totalElements(shape));
         if(isContiguous()){
            return view(newShape);   // free
        } else {
            Tensor contiguous = makeContiguous();  // physical copy using odometer
            return contiguous.view(newShape);      // then view
        }
    }
    int64_t getSize() const{
        int64_t sum=1;
        for(const auto& sh:shape) sum*=sh;
        return sum;
    }
    int64_t totalElements(std::span<const int64_t> s) const {
        int64_t total = 1;
        for (auto dim : s) total *= dim;
        return total;
    }
    // Overload for your internal shape vector
    int64_t totalElements(const std::vector<int64_t>& s) const {
        int64_t total = 1;
        for (auto dim : s) total *= dim;
        return total;
    }
    Tensor  makeContiguous() const{
        Tensor res(shape);
        
        std::vector<int64_t> id(res.shape.size(), 0);
   
        for(size_t i=0;i<res.getSize();i++){
            (*res.data)[i]=this->getElement(id);

            for(int64_t j=res.shape.size()-1;j>=0;j--){
                id[j]+=1;
                if(id[j]<shape[j]) break;
                id[j]=0;
                
            }
        }
        return res;
    }
    
    bool isContiguous() const{
        auto expected = computeStrides(shape);
        return expected == strides;
    }
    //this one for the user so it will tell before the user runs it     
    template <typename... Args>
    T& getElement(Args... indices) {
        // 1. Safety Checks
        static_assert(sizeof...(Args) > 0, "Must provide indices");
        static_assert((std::is_integral_v<Args> && ...), "Indices must be integers");
        
        // In a real pro library, we'd also check if sizeof...(Args) == shape.size()
        // but that requires a 'runtime' check since shape.size() isn't known at compile time.
        if (sizeof...(Args) != shape.size()) {
            throw std::invalid_argument(
                "Tensor Rank Mismatch: Expected " + std::to_string(shape.size()) +
                " indices, but got " + std::to_string(sizeof...(Args)) + "."
            );
        }
        int idx_array[] = { static_cast<int>(indices)... };
        int flat_idx = 0;
        for (int i = 0; i < sizeof...(indices); ++i) {
            flat_idx += idx_array[i] * strides[i];
        }
        
        return (*data)[flat_idx];
    }
    //this one for the internal calculations
    T& getElement(const std::vector<int64_t>& idx) {
        if(idx.size() != shape.size())
            throw std::invalid_argument("Rank mismatch");

        int64_t flat_idx = 0;
        for(size_t i = 0; i < idx.size(); i++)
            flat_idx += idx[i] * strides[i];

        return (*data)[flat_idx];
    }
    const T& getElement(const std::vector<int64_t>& idx) const {
        if(idx.size() != shape.size())
            throw std::invalid_argument("Rank mismatch");

        int64_t flat_idx = 0;
        for(size_t i = 0; i < idx.size(); i++)
            flat_idx += idx[i] * strides[i];

        return (*data)[flat_idx];
    }
    template <typename... Args>
    T& operator()(Args... indices) noexcept{
        size_t i=0;
        // CRITICAL FIX: Start at the offset so Views work correctly!
        size_t flat_idx = this->offset;

        ((flat_idx += static_cast<size_t>(indices) * strides[i++]), ...); // this beauty called Binary Comma Fold Expression
        
        return (*data)[flat_idx];
    }
    template <typename... Args>
    const T& operator()(Args... indices) const {
        size_t i = 0;
        // CRITICAL FIX: Start at the offset so Views work correctly!
        size_t flat_idx = this->offset;
        ((flat_idx += static_cast<size_t>(indices) * strides[i++]), ...);
        return (*data)[flat_idx];
    }
    Tensor<T>& operator=(const Tensor<T>& other){
        if (this == &other) return *this;
        if (!checkShape(other)) {
            throw std::invalid_argument("Shapes must match to assign an entire row/batch!");
        }
        // 3. Calculate total elements in this specific view
        size_t total_elements = 1;
        for (size_t dim : shape) total_elements *= dim;

        // 4. Safely copy the raw numbers from 'other' into 'this' View's shared RAM
        T* dest_ptr = this->data->data() + this->offset;
        const T* src_ptr = other.data->data() + other.offset;
        // Since operator[] only drops the outermost dimension, the memory block 
        // is guaranteed to be contiguous! A simple flat loop works perfectly.
        for (size_t i = 0; i < total_elements; i++) {
            dest_ptr[i] = src_ptr[i];
        }
        return *this;
    }
    bool checkShape(const Tensor<T>& other) const{
        if(shape.size()!=other.shape.size()) throw std::invalid_argument("Dimension must match for Sub/Add");
        for(size_t i=0;i<shape.size();i++){
            if(shape[i]!=other.shape[i]) return false;
        }
        return true;
    }
    // Inside your Tensor class:
    Tensor<T> getMatrix(size_t batch_index) const {
        assert(shape.size() >= 2);

        int64_t rows = shape[shape.size() - 2];
        int64_t cols = shape[shape.size() - 1];
        int64_t matrix_size = rows * cols;

        Tensor<T> mat({rows, cols});
        mat.data = this->data;                           // share, no copy
        mat.offset = batch_index * matrix_size;
        return mat;
    }
    Tensor<T> getMatrix(size_t batch_index, const std::vector<int64_t>& res_shape,const std::vector<int64_t>& padded_shape) const{
        assert(shape.size() >= 2);
        int64_t rows = shape[shape.size() - 2];
        int64_t cols = shape[shape.size() - 1];
        int64_t matrix_size = rows * cols;
        int64_t actual_offset = 0;
        int64_t stride = matrix_size;
        int64_t remaining_i=batch_index;

        for (int d = (int)res_shape.size() - 3; d >= 0; d--){
            int64_t coord = remaining_i % res_shape[d];
            remaining_i /= res_shape[d]; 

            if(padded_shape[d]!=1){
                actual_offset += coord * stride;
                stride *= padded_shape[d];
            }
        }
        Tensor<T> mat({rows, cols});
        mat.data = this->data;                           // share, no copy
        mat.offset = actual_offset;
        return mat;
     }

    //Basic operators
    Tensor<T> operator*(const Tensor& other)const{
        size_t dimension=shape.size();
        size_t dimensionOther=other.shape.size();
        //dimension can be 1 as well 
        //we do'nt check if the dimensions are zero 
        //we want to add 1s in the sahep of one of them might need some mock 
        
        
        if(shape[dimension-1]!=other.shape[dimensionOther-2]) throw std::invalid_argument("Last 2 dimension must be KxN * NxR");
        if(dimension==2&&dimensionOther==2) return matMul2d(other);

        bool exactMatch=checkMatMul(other);
        bool broadcastable=false;
        std::vector<int64_t> padded_shape_A = this->shape;
        std::vector<int64_t> padded_shape_B = other.shape;
        if(!exactMatch){


            // If A is smaller, pad 1s onto the left side of A
            while (padded_shape_A.size() < padded_shape_B.size()) {
                padded_shape_A.insert(padded_shape_A.begin(), 1);
            }

            // If B is smaller, pad 1s onto the left side of B
            while (padded_shape_B.size() < padded_shape_A.size()) {
                padded_shape_B.insert(padded_shape_B.begin(), 1);
            }
            broadcastable=checkBroadCastable(padded_shape_A,padded_shape_B);

        }
        if(!exactMatch && !broadcastable) throw std::invalid_argument("Other dimension must match or need to be broadcastable");

        if(exactMatch){
            Tensor<T> resTens(matMulShape(other));
            size_t rank = resTens.shape.size();
            int64_t M = resTens.shape[rank - 2]; // The rows of the final 2D matrix
            int64_t N = resTens.shape[rank - 1]; // The cols of the final 2D matrix
            int64_t sz = M * N;
             T* val_data=resTens.data->data()+resTens.offset;
            int64_t total_batches=countBatches();
            for(int64_t i=0;i<total_batches;i++){
                Tensor<T> matrix_A = this->getMatrix(i);
                Tensor<T> matrix_B = other.getMatrix(i);
                T* batch_destination = val_data + (i * sz);
                matrix_A.matMul2d_direct(matrix_B, batch_destination);
                // Tensor<T> result_matrix = matrix_A.matMul2d(matrix_B);
                // int64_t sz=result_matrix.getSize();;
                // T* res_raw = result_matrix.data->data();
                // for(int64_t k = 0;k<sz;k++){
                //     val_data[i*sz+k]=res_raw[k];
                // }
            }
            return resTens;
        }
        
        Tensor<T> resTens(matMulShapePadded(padded_shape_A,padded_shape_B));
        T* val_data=resTens.data->data()+resTens.offset;
        // 1. Find the last two dimensions of the result tensor
        size_t rank = resTens.shape.size();
        int64_t M = resTens.shape[rank - 2]; // The rows of the final 2D matrix
        int64_t N = resTens.shape[rank - 1]; // The cols of the final 2D matrix
        int64_t sz = M * N;
        int64_t total_batches=resTens.countBatches();
        for(int64_t i=0;i<total_batches;i++){
            Tensor<T> matrix_A = this->getMatrix(i, resTens.shape,padded_shape_A);
            Tensor<T> matrix_B = other.getMatrix(i,resTens.shape,padded_shape_B);
            T* batch_destination = val_data + (i * sz);
            //this cost too mcu we create a brand new result_matrix
            // Tensor<T> result_matrix = matrix_A.matMul2d(matrix_B);
            // int64_t sz=result_matrix.getSize();;

            matrix_A.matMul2d_direct(matrix_B, batch_destination);
        }
          
        return resTens;
        
    } 

     bool checkBroadCastable(std::vector<int64_t>A, std::vector<int64_t>B) const {
        if (A.size() != B.size()) return false;
        for (size_t i = 0; i < A.size() - 2; i++) {
            if (A[i] != B[i] && A[i] != 1 && B[i] != 1) {
                return false;
            }
        }
        return true;
    }
    // bool checkBroadCastable(const Tensor& other) const {
    //     if (shape.size() != other.shape.size()) return false;
    //     for (size_t i = 0; i < dimension - 2; i++) {
    //         if (shape[i] != other.shape[i] && shape[i] != 1 && other.shape[i] != 1) {
    //             return false;
    //         }
    //     }
    //     return true;
    // }

    bool checkMatMul(const Tensor& other) const{
        
        size_t dimension=shape.size();
        if(dimension!=other.shape.size())return false;
        for(size_t i=0;i<dimension-2;i++){
            if(shape[i]!=other.shape[i]) return false;
        }
        return true;
    }
    std::vector<int64_t> matMulShape(const Tensor& other)const{
        std::vector<int64_t> newShape=shape;
        newShape[newShape.size()-1]=other.shape[newShape.size()-1];
        return newShape;
    }
    std::vector<int64_t> matMulShapePadded(const std::vector<int64_t>& A, const std::vector<int64_t>& B)const{
        std::vector<int64_t> newShape=A;
        for(size_t i=0;i<A.size()-2;i++){
            if(newShape[i]<B[i]){
                newShape[i]=B[i];
            }
        }
        newShape[newShape.size()-1]=B[newShape.size()-1];
        return newShape;
    }
    int64_t countBatches()const{
        int64_t mult=1;
        for(size_t i=0;i<shape.size()-2;i++){
            mult*=shape[i];
        }
        return mult;
    }
    Tensor<T> matMul2d(const Tensor<T>& other)const{
        if(shape.size()!=2 || other.shape.size()!=2) throw std::invalid_argument("Dimensions must be 2");
        if(shape[1]!=other.shape[0]) throw std::invalid_argument("Matrix multiplication must be KxN * NxR");
        size_t M = shape[0];
        size_t K = shape[1];
        size_t N = other.shape[1];
        Tensor<T> res(std::vector<int64_t>{M, N});
        const T* a_data = this->data->data() + this->offset;
        const T* b_data = other.data->data() + other.offset;
        T* res_data=res.data->data();
        for(size_t i=0;i<M;i++){
            for(size_t j=0;j<K;j++){
                T a_val = a_data[i * K + j];//we are absically skipping tzhe first row so absically the K chracter
                for(size_t k=0;k<N;k++){
                    res_data[i*N+k]+=a_val*b_data[j*N+k];// 
                }
            }
        }
        return res;

    }
//for learning because of the res new amtrix taht would cost a lot when  we would sue this for llms
// NEW FUNCTION: Writes directly to pre-allocated RAM!
    void matMul2d_direct(const Tensor<T>& other, T* dest_ptr) const {
        if(shape.size() != 2 || other.shape.size() != 2) throw std::invalid_argument("Dimensions must be 2");
        if(shape[1] != other.shape[0]) throw std::invalid_argument("Matrix multiplication must be KxN * NxR");
        
        size_t M = shape[0];
        size_t K = shape[1];
        size_t N = other.shape[1];
        
        // Notice: NO Tensor<T> res(...) allocation here!
        
        const T* a_data = this->data->data() + this->offset;
        const T* b_data = other.data->data() + other.offset;
        
        // Initialize the specific block of destination memory to 0
        for(size_t i = 0; i < (M * N); i++) {
            dest_ptr[i] = 0;
        }

        // Your brilliantly optimized cache-friendly loops
        for(size_t i = 0; i < M; i++) {
            for(size_t j = 0; j < K; j++) {
                T a_val = a_data[i * K + j]; 
                for(size_t k = 0; k < N; k++) {
                    // Writing straight into the final batch memory!
                    dest_ptr[i * N + k] += a_val * b_data[j * N + k]; 
                }
            }
        }
    }

    Tensor<T>& operator+=(const Tensor<T>& other) {
        if (this->getSize() != other.getSize()) {
        throw std::invalid_argument("Tensor sizes must match for addition.");
    }
        if(!checkShape(other)){
             throw std::invalid_argument("Tensor sizes must match for addition.");
        }
        for(size_t i=0;i<data->size();i++){
            (*this->data)[i] += (*other.data)[i];
        }
        return *this;
    }
    Tensor<T> operator+(const Tensor<T>& other) const{
        Tensor<T> res(shape);    // constructor allocates a brand-new vector
        *res.data   = *data;     // deep copy the contents
        res.strides = strides;   // preserve strides (e.g. if non-contiguous)
        res+=other;
        return res;
    }
    Tensor<T>& operator+=(const T value ) {
        for(size_t i=0;i<data->size();i++){
            (*this->data)[i] += value;
        }
        return *this;
    }
    Tensor<T> operator+(const T value )const {
        Tensor<T> res(shape);    // constructor allocates a brand-new vector
        *res.data   = *data;     // deep copy the contents
        res.strides = strides;   // preserve strides (e.g. if non-contiguous)
        for(size_t i=0;i<data->size();i++){
            (*res.data)[i] += value;
        }
        return res;
    }
    Tensor<T>& operator-=(const Tensor<T>& other) {
        if (this->getSize() != other.getSize()) {
        throw std::invalid_argument("Tensor sizes must match for addition.");
    }
        if(!checkShape(other)){
             throw std::invalid_argument("Tensor sizes must match for subtraction.");
        }
        for(size_t i=0;i<data->size();i++){
            (*this->data)[i] -= (*other.data)[i];
        }
        return *this;
    }
    Tensor<T> operator-(const Tensor<T>& other)const {
        Tensor<T> res(shape);    // constructor allocates a brand-new vector
        *res.data   = *data;     // deep copy the contents
        res.strides = strides;   // preserve strides (e.g. if non-contiguous)
        res-=other;
        return res;
    }
    Tensor<T> sum(int dimension)const{
        if(dimension>shape.size()-1){
            throw std::invalid_argument("Must be in the interval of [0, N-1]");
        }
        int64_t st=strides[dimension];
        std::vector<int64_t> newShape;
        for(size_t i=0;i<shape.size();i++){
            if(i!=dimension){
                newShape.push_back(shape[i]);
            }
        }
        Tensor<T> res(newShape);
        T* res_raw = res.data->data() + res.offset;
        const T* in_raw = this->data->data() + this->offset;
        if (dimension == shape.size() - 1) {
            int64_t chunk_size = shape[dimension]; // How many numbers are side-by-side
            size_t out_idx = 0;
            
            // Just walk straight down the flat RAM!
            for (size_t i = 0; i < this->getSize(); i += chunk_size) {
                T current_sum = 0;
                for (int64_t k = 0; k < chunk_size; k++) {
                    current_sum += in_raw[i + k];
                }
                res_raw[out_idx++] = current_sum;
            }
            return res;
        }

        for(size_t i = 0; i < res.getSize(); i++) {
            res_raw[i] = 0;
        }
        for(size_t i = 0; i < this->getSize(); i++) {
    
            // We need to figure out the exact destination 'out_id' for this specific 'i'
            size_t out_id = 0;
            size_t out_dim = 0;
            size_t remaining_i = i;

            // Turn our flat 'i' into N-Dimensional coordinates
            for(size_t d = 0; d < shape.size(); d++) {
                size_t coord = remaining_i / this->strides[d];
                remaining_i %= this->strides[d];

                // If this is NOT the dimension we are destroying...
                // map this coordinate to the Result tensor!
                if(d != dimension) {
                    out_id += coord * res.strides[out_dim];
                    out_dim++;
                }
            }

            // Boom. Push the data into the correct bucket.
            res_raw[out_id] += in_raw[i];
        }
        return res;
    }
    Tensor<T> mean(int dimension) const{
        Tensor<T> result=Tensor::sum(dimension);
        int64_t sh=shape[dimension];
        T* res_raw = result.data->data() + result.offset;
        for(size_t i=0;i<result.getSize();i++){
            res_raw[i]/=static_cast<T>(sh);
        }
        return result;
    }  
    Tensor<T> max(int dimension ){
        //concept we wnat to eliminate the full dimension but not with summing them but comparing them
        if(dimension>shape.size()-1){
            throw std::out_of_range("Must be in the interval of [0, N-1]");
        }
        std::vector<int64_t> newShape;
        for(size_t i=0;i<shape.size();i++){
            if(i!=dimension){
                newShape.push_back(shape[i]);
            }
        }
        Tensor<T> res(newShape);
        T* res_raw=res.data->data()+res.offset;
        const T* val=this->data->data()+this->offset;
        int total_el=1;
        for(size_t i=0;i<newShape.size();i++){
            total_el*=newShape[i];
        }
        for(size_t i=0;i<total_el;i++){
            res_raw[i]=std::numeric_limits<T>::lowest();
        }

        for(size_t i=0;i<getSize();i++){
            int dim=0;
            int id=0;
            int stride=1;
            int remaining_i=i;

            for(size_t d =0;d<shape.size();d++){
                int coord =remaining_i/strides[d];
                remaining_i%=strides[d];


                if(dimension!=d){
                    id+=coord*res.strides[dim];

                    dim++;
                }
            }
            if(res_raw[id]<val[i]){
                res_raw[id]=val[i];
            }
        }
        return res;

    }
    // Tensor<T> bmm (const Tensor<T> other) const{
    //     if(shape.size()!=3 ||other.shape.size()!=3) std::invalid_argument("Tensors msut be rank 3");
    //     if(shape[0]!=other.shape[0]) std::invalid_argument("Batch dimensions must match for bmm");
    //     if(shape[2]==other.shape[1]) std::invalid_argument("Inner 2d amtrix msut have mutiplciation KxN * NxM");
    //     Tensor<T> result(std::vector<int64_t>({shape[0],shape[1], other.shape[2]}));

    //     const T* A_raw=this->data->data()+this->offset;
    //     const T* B_raw=other.data->data()+other.offset;
    //     T* C_raw= result.data->data()+result.offset;
    //     int64_t res_size=1;
    //     for(const auto& d:result.shape){
    //         res_size*=d;
    //     }
    //     for(size_t i=0;i<res_size;i++){
    //         C_raw[i]=0;
    //     }
    //     for(size_t b=0;b<result.shape[0];b++){
    //         for(size_t r=0;r<result.shape[1];r++){
    //             for(size_t c=0;r<result.shape[2];c++){
    //                 for(size_t k=0;r<shape[2];k++){
    //                     index_A = offset_A + (b * A.strides[0]) + (m * A.strides[1]) + (k * A.strides[2]);
    //                     index_B = offset_B + (b * B.strides[0]) + (k * B.strides[1]) + (n * B.strides[2]);
    //                     index_C = offset_C + (b * C.strides[0]) + (m * C.strides[1]) + (n * C.strides[2]);
    //                     C_raw[index_C] += A_raw[index_A] * B_raw[index_B]
    //                 }
    //             }
    //         }
    //     }
    //     return result;
    // }

    Tensor<T> operator[](size_t index) const{

        if (shape.size() == 0) throw std::out_of_range("Cannot index a 0D scalar.");
        if (index >= shape[0]) throw std::out_of_range("Index out of bounds.");
        std::vector<int64_t> newShape(shape.begin()+1, shape.end());
        int64_t new_offset = this->offset + (index * this->strides[0]);
        Tensor<T> view(newShape,this->data,new_offset);

        return view;
    }
    Tensor<T> PGEMM(const Tensor<T>& other )const{
        
        unsigned int num_threads = std::thread::hardware_concurrency();
        unsigned int total_rows=this->shape[0];
        unsigned int base_chunk = total_rows / num_threads;
        
        Tensor<T> C(std::vector<int64_t>{total_rows,other.shape[1]});
        std::vector<std::thread> threads;
        for(size_t i=0;i<num_threads;i++){
            int start_row=i*base_chunk;
            int end_row= (num_threads-1)==i ? total_rows:(start_row + base_chunk);
            Tensor<T> C_chunk =C.getRows(start_row, end_row);
            Tensor<T> A_chunk =this->getRows(start_row, end_row);

            threads.push_back(std::thread(worker_gemm, C_chunk,A_chunk, other));
        }

        for (auto& t : threads) {
            t.join();
        }
        return C;
    }
    Tensor<T> getRows(int start,int end )const{
        int dimension=shape.size();
        int num_rows = shape[dimension - 2];
        if(start < 0 || end < start || end > num_rows) throw std::out_of_range("Index out of bounds.");
        int nOffset=this->offset +(start * strides[dimension - 2]);
        int rows_size=end-start;
        std::vector<int64_t> s{rows_size, shape[dimension-1]};
        Tensor<T> rows(s,data,nOffset);
        return rows;
    }
    // A B C are light weight they all jsut like pointers to the real data
    static void worker_gemm(Tensor<T> C, const Tensor<T> A, const Tensor<T> B) {
        // Inside here, the thread just loops from start_row to end_row.
        // It reads its rows from A.
        // It reads ALL of B.
        // It writes only to its rows in C.
        T* A_raw = A.data->data()+A.offset;
        T* B_raw = B.data->data()+B.offset;
        T* C_raw = C.data->data()+C.offset;
        int N=A.shape[0];
        int K=A.shape[1];
        int M=B.shape[1];
        //NxK * KxM
        for(size_t i=0;i<N;i++){
            for(size_t k=0;k<K;k++){
                T a_val = A_raw[i * K + k];
                for(size_t r=0;r<M;r++){
                    C_raw[i*M+r]+=a_val *B_raw[k * M + r];
                }

            }
        }
        // const int BLOCK = 64; 

    //     for (int ii = 0; ii < N; ii += BLOCK) {
    //         for (int kk = 0; kk < K; kk += BLOCK) {
    //             for (int jj = 0; jj < M; jj += BLOCK) {
                    
    //                 // Fix the 'j' loop here!
    //                 for (int i = ii; i < std::min(ii + BLOCK, N); ++i) {
    //                     for (int k = kk; k < std::min(kk + BLOCK, K); ++k) {
    //                         float a_val = A_raw[i * K + k];
    //                         // Corrected: j < std::min(...), not jj
    //                         for (int j = jj; j < std::min(jj + BLOCK, M); ++j) {
    //                             C_raw[i * M + j] += a_val * B_raw[k * M + j];
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }

};

#endif
