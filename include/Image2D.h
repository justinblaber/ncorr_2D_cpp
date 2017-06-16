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
                
        // Static factory methods --------------------------------------------//
        static Image2D load(std::ifstream&);
            
        // Interface functions -----------------------------------------------//
        friend std::ostream& operator<<(std::ostream&, const Image2D&); 
        friend void imshow(const Image2D&, difference_type delay = -1);  
        friend bool isequal(const Image2D&, const Image2D&);
        friend void save(const Image2D&, std::ofstream&);
        
        // Access ------------------------------------------------------------//
        std::string get_filename() const { return *filename_ptr; }
        Array2D<double> get_gs() const; // Returns image as double precision grayscale array with values from 0 - 1.
        
    private:
        std::shared_ptr<std::string> filename_ptr; // immutable
};

}

#endif	/* IMAGE2D_H */