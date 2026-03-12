#ifndef DATAMANGER_HPP
#define DATAMANGER_HPP

template <typename T>
class DataManager{
    std::vector<int> shape;
    std::vector<int> strides;
    std::shared_ptr<std::vector<T>> data;
public:
    DataManager(std::vector<T>() s):shape(s), data(std::make_shared<std::vector<T>>()){
        computeStrides();
        // Calculate total size and resize data
        data = std::make_shared<std::vector<T>>();
        size_t total_size = 1;
        for(int dim : shape) total_size *= dim;
        data->resize(total_size);
    }
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
};
#endif