/* 
 * File:   Image2D.h
 * Author: justin
 *
 * Created on January 28, 2015, 3:10 PM
 */

#include "Image2D.h"

namespace ncorr {

// Static factory methods ----------------------------------------------------//
Image2D Image2D::load(std::ifstream &is) {
    // Form empty Image2D then fill in values in accordance to how they are saved
    Image2D img;
    
    // Load length
    difference_type length = 0;
    is.read(reinterpret_cast<char*>(&length), std::streamsize(sizeof(difference_type)));
    
    // Allocate new string
    img.filename_ptr = std::make_shared<std::string>(length,' ');
        
    // Read data
    is.read(const_cast<char*>(img.filename_ptr->c_str()), std::streamsize(length));
    
    return img;
}

// Operations interface ------------------------------------------------------//
std::ostream& operator<<(std::ostream &os, const Image2D &img) {
    os << *img.filename_ptr;
    
    return os;
}

void imshow(const Image2D &img, Image2D::difference_type delay) { 
    imshow(img.get_gs(),delay);
}  

bool isequal(const Image2D &img1, const Image2D &img2) {
    return *img1.filename_ptr == *img2.filename_ptr;
}

void save(const Image2D &img, std::ofstream &os) {    
    typedef Image2D::difference_type                            difference_type;
    
    // Save length -> image name
    difference_type length = img.filename_ptr->size();
    os.write(reinterpret_cast<const char*>(&length), std::streamsize(sizeof(difference_type)));
    os.write(img.filename_ptr->c_str(), std::streamsize(img.filename_ptr->size()));
}

// Access --------------------------------------------------------------------//
Array2D<double> Image2D::get_gs() const {
    // get_gs() uses opencv's imread() function; must convert Mat to Array2D type.
    cv::Mat cv_img = cv::imread((*filename_ptr), cv::IMREAD_GRAYSCALE);
    if (!cv_img.data) {
        throw std::invalid_argument("Image file : " + *filename_ptr + " cannot be found or read.");
    }           

    // Images will be read as 8-bit grayscale values; convert these to double 
    // precision with values ranging from 0 - 1.
    Array2D<double> A(cv_img.rows,cv_img.cols);    
    double conversion = 1/255.0;
    for (difference_type p1 = 0; p1 < cv_img.cols; ++p1) {
        for (difference_type p2 = 0; p2 < cv_img.rows; ++p2) {
            A(p2,p1) = cv_img.data[p1 + p2*cv_img.cols] * conversion;
        }
    }           

    return A;
}

}