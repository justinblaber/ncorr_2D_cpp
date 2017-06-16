/* 
 * File:   Disp2D.h
 * Author: justin
 *
 * Created on May 12, 2015, 12:36 PM
 */

#ifndef DISP2D_H
#define	DISP2D_H

#include "Array2D.h"
#include "Image2D.h"
#include "ROI2D.h"
#include "Data2D.h"

namespace ncorr {

namespace details {
    class Disp2D_nlinfo_interpolator; 
}

class Disp2D final { 
// ---------------------------------------------------------------------------//
// Disp2D is a class for 2D displacements. -----------------------------------//
// ---------------------------------------------------------------------------//
public:    
    typedef Data2D::difference_type                             difference_type;    
    typedef Data2D::coords                                               coords; 
    typedef details::Disp2D_nlinfo_interpolator             nlinfo_interpolator; 
    
    // Rule of 5 and destructor ----------------------------------------------//    
    Disp2D() noexcept = default;
    Disp2D(const Disp2D&) = default;
    Disp2D(Disp2D&&) noexcept = default;
    Disp2D& operator=(const Disp2D&) = default;
    Disp2D& operator=(Disp2D&&) = default; 
    ~Disp2D() noexcept = default;
    
    // Additional constructors -----------------------------------------------//
    Disp2D(Array2D<double> v, Array2D<double> u, const ROI2D &roi, difference_type scalefactor) : // r-value
         v(std::move(v), roi, scalefactor), u(std::move(u), roi, scalefactor) { }
        
    // Static factory methods ------------------------------------------------//
    static Disp2D load(std::ifstream&);

    // Operators interface ---------------------------------------------------//
    friend std::ostream& operator<<(std::ostream&, const Disp2D&); 
    friend void imshow(const Disp2D&, difference_type delay = -1);  
    friend bool isequal(const Disp2D&, const Disp2D&);
    friend void save(const Disp2D&, std::ofstream&);
        
    // Access ----------------------------------------------------------------//
    // Note that Disp2D is immutable, so all access is const.    
    difference_type data_height() const { return v.data_height(); }
    difference_type data_width() const { return v.data_width(); }
    const Data2D& get_v() const { return v; } 
    const Data2D& get_u() const { return u; } 
    const ROI2D& get_roi() const { return v.get_roi(); }   
    difference_type get_scalefactor() const { return v.get_scalefactor(); }
    
    // Interpolator ----------------------------------------------------------//
    nlinfo_interpolator get_nlinfo_interpolator(difference_type, INTERP) const;    
                
    // Utility ---------------------------------------------------------------//
    std::string size_string() const { return v.size_string(); }   
    std::string size_2D_string() const { return v.size_2D_string(); }   
    
private:        
    Data2D v;   // Immutable - Data2D has pointer semantics
    Data2D u;   // Immutable - Data2D has pointer semantics
};
  
namespace details {    
    class Disp2D_nlinfo_interpolator final {    
        public:      
            typedef Disp2D::difference_type                     difference_type;   
            typedef Disp2D::coords                                       coords;
            
            friend Disp2D;
            
            // Rule of 5 and destructor --------------------------------------//
            Disp2D_nlinfo_interpolator() noexcept = default;
            Disp2D_nlinfo_interpolator(const Disp2D_nlinfo_interpolator&) = default;
            Disp2D_nlinfo_interpolator(Disp2D_nlinfo_interpolator&&) = default;
            Disp2D_nlinfo_interpolator& operator=(const Disp2D_nlinfo_interpolator&) = default;  
            Disp2D_nlinfo_interpolator& operator=(Disp2D_nlinfo_interpolator&&) = default;
            ~Disp2D_nlinfo_interpolator() noexcept = default;
            
            // Additional Constructors ---------------------------------------//            
            Disp2D_nlinfo_interpolator(const Disp2D &disp, difference_type region_idx, INTERP interp_type) :
                v_interp(disp.get_v().get_nlinfo_interpolator(region_idx,interp_type)), u_interp(disp.get_u().get_nlinfo_interpolator(region_idx,interp_type)) { }
             
            // Access methods ------------------------------------------------//
            std::pair<double,double> operator()(double p1, double p2) const { return { v_interp(p1,p2), u_interp(p1,p2) }; }
            std::pair<const Array2D<double>&,const Array2D<double>&> first_order(double p1, double p2) const { return { v_interp.first_order(p1,p2), u_interp.first_order(p1,p2) }; }
            
        private:       
            Data2D::nlinfo_interpolator v_interp; // must have copy of interpolator
            Data2D::nlinfo_interpolator u_interp; // must have copy of interpolator
    };    
}

}

#endif	/* DISP2D_H */

