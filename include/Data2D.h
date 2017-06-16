/* 
 * File:   Data2D.h
 * Author: justin
 *
 * Created on February 28, 2015, 8:57 PM
 */

#ifndef DATA2D_H
#define	DATA2D_H

#include "Array2D.h"
#include "Image2D.h"
#include "ROI2D.h"

namespace ncorr {

namespace details {
    class Data2D_nlinfo_interpolator; 
}

class Data2D final { 
// -------------------------------------------------------------------------- //
// -------------------------------------------------------------------------- //
// Data2D encompasses everything associated with a 2D Data array. It          //
// supports:                                                                  //
//  1) Having a scalefactor, as some 2D data plots can be "reduced" to save   //
//     space or computational savings.                                        //
//  2) Having a region of interest, as not all 2D Data are "full." This       //
//     supports ROI2D based interpolation.                                    //                                                          //   
// -------------------------------------------------------------------------- //
// -------------------------------------------------------------------------- //
public:    
    typedef std::ptrdiff_t                                      difference_type;    
    typedef std::pair<difference_type,difference_type>                   coords;   
    typedef details::Data2D_nlinfo_interpolator             nlinfo_interpolator; 
    
    // Rule of 5 and destructor ----------------------------------------------//    
    Data2D() noexcept : scalefactor() { }
    Data2D(const Data2D&) = default;
    Data2D(Data2D&&) noexcept = default;
    Data2D& operator=(const Data2D&) = default;
    Data2D& operator=(Data2D&&) = default; 
    ~Data2D() noexcept = default;
    
    // Additional constructors -----------------------------------------------//
    Data2D(Array2D<double>, const ROI2D&, difference_type); // r-value
    Data2D(Array2D<double>, const ROI2D&); // r-value
    Data2D(Array2D<double>, difference_type); // r-value
    explicit Data2D(Array2D<double>); // r-value
    
    // Static factory methods ------------------------------------------------//
    static Data2D load(std::ifstream&);
        
    // Operators interface ---------------------------------------------------//
    friend std::ostream& operator<<(std::ostream&, const Data2D&); 
    friend void imshow(const Data2D&, difference_type delay = -1);  
    friend bool isequal(const Data2D&, const Data2D&);
    friend void save(const Data2D&, std::ofstream&);
       
    // Access ----------------------------------------------------------------//
    // Note that Data2D is immutable, so all access is const.    
    difference_type data_height() const { return A_ptr->height(); }
    difference_type data_width() const { return A_ptr->width(); }
    const Array2D<double>& get_array() const { return *A_ptr; } 
    const ROI2D& get_roi() const { return roi; } 
    difference_type get_scalefactor() const { return scalefactor; }
    
    // Interpolator ----------------------------------------------------------//
    nlinfo_interpolator get_nlinfo_interpolator(difference_type, INTERP) const;
    
    // Utility ---------------------------------------------------------------//
    std::string size_string() const { return std::to_string(A_ptr->size()); }   
    std::string size_2D_string() const { return "(" + std::to_string(data_height()) + "," + std::to_string(data_width()) + ")"; }   
    
private:    
    // Utility ---------------------------------------------------------------//
    void chk_scalefactor() const;
    void chk_data_roi_same_size() const;
    
    difference_type scalefactor;
    ROI2D roi;                                  // immutable - ROI2D already has pointer semantics
    std::shared_ptr<Array2D<double>> A_ptr;     // immutable
};

namespace details {
    class Data2D_nlinfo_interpolator final {    
        public:      
            typedef Data2D::difference_type                     difference_type;   
            typedef Data2D::coords                                       coords;
            
            friend Data2D;
            
            // Rule of 5 and destructor --------------------------------------//
            Data2D_nlinfo_interpolator() noexcept : scalefactor(), nlinfo_top(), nlinfo_left() { }
            Data2D_nlinfo_interpolator(const Data2D_nlinfo_interpolator&) = default;
            Data2D_nlinfo_interpolator(Data2D_nlinfo_interpolator&&) = default;
            Data2D_nlinfo_interpolator& operator=(const Data2D_nlinfo_interpolator&) = default;  
            Data2D_nlinfo_interpolator& operator=(Data2D_nlinfo_interpolator&&) = default;
            ~Data2D_nlinfo_interpolator() noexcept = default;
            
            // Additional Constructors ---------------------------------------//            
            Data2D_nlinfo_interpolator(const Data2D&, difference_type, INTERP);
             
            double operator()(double p1, double p2) const { return sub_data_interp(p1_unscaled(p1), p2_unscaled(p2)); }
            const Array2D<double>& first_order(double p1, double p2) const { 
                const auto &fo_unscaled = sub_data_interp.first_order(p1_unscaled(p1), p2_unscaled(p2)); 
                first_order_buf(0) = fo_unscaled(0);                   // value - do not modify
                first_order_buf(1) = fo_unscaled(1) / scalefactor;     // p1 gradient - must scale 
                first_order_buf(2) = fo_unscaled(2) / scalefactor;     // p2_gradient - must scale
                
                return first_order_buf;
            }
            
        private:
            // Access methods ------------------------------------------------//
            double p1_unscaled(double p1) const { return (p1 / scalefactor) - nlinfo_top + border; }
            double p2_unscaled(double p2) const { return (p2 / scalefactor) - nlinfo_left + border; } 
            
            std::shared_ptr<Array2D<double>> sub_data_ptr; // immutable
            Array2D<double>::interpolator sub_data_interp; // must have copy of interpolator
            mutable Array2D<double> first_order_buf;       // have copy
            difference_type scalefactor;
            difference_type nlinfo_top;
            difference_type nlinfo_left;
            difference_type border = 20;
    };       
}

}

#endif	/* DATA2D_H */