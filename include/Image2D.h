/* 
 * File:   Image2D.h
 * Author: justin
 *
 * Created on January 28, 2015, 3:10 PM
 */

#ifndef IMAGE2D_H
#define	IMAGE2D_H

#include "Array2D.h"

namespace ncorr {     
    
class Image2D final { 
    public:                 
        typedef std::ptrdiff_t                                  difference_type; 
        
        // Rule of 5 and destructor ------------------------------------------//        
        Image2D() = default;
        Image2D(const Image2D&) = default;
        Image2D(Image2D&&) noexcept = default;
        Image2D& operator=(const Image2D&) = default;
        Image2D& operator=(Image2D&&) = default; 
        ~Image2D() noexcept = default;
        
        // Additional Constructors -------------------------------------------//
        // Allow implicit conversion        
        Image2D(std::string filename) : filename_ptr(std::make_shared<std::string>(std::move(filename))) { } // by-value
        Image2D(const Array2D<double> &array_data) : image_data(array_data),filename_ptr(std::make_shared<std::string>("(none)")) { } // by-value
                
        // Static factory methods --------------------------------------------//
        static Image2D load(std::ifstream&);
            
        // Interface functions -----------------------------------------------//
        friend std::ostream& operator<<(std::ostream&, const Image2D&); 
        // sdh4 09/18/19 move imshow() contents into header to accommodate
        // http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#136
        // (See "Proposed resolution" that any frend declaration with a default
        // argument expression must be a definition)
        friend void imshow(const Image2D &img, difference_type delay = -1) {
	  if (img.image_data.get_pointer()) {
	    imshow(img.image_data,delay);
	  } else {
	    imshow(img.get_gs(),delay);
	  }
	  
	}
        friend bool isequal(const Image2D&, const Image2D&);
        friend void save(const Image2D&, std::ofstream&);
        
        // Access ------------------------------------------------------------//
        std::string get_filename() const { return *filename_ptr; }
        Array2D<double> get_gs() const; // Returns image as double precision grayscale array with values from 0 - 1.
        
    private:
        std::shared_ptr<std::string> filename_ptr; // immutable
        Array2D<double> image_data; // immutable
};

}

#endif	/* IMAGE2D_H */
