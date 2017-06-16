/* 
 * File:   Strain2D.h
 * Author: justin
 *
 * Created on June 7, 2015, 10:14 PM
 */

#ifndef STRAIN2D_H
#define	STRAIN2D_H

#include "Array2D.h"
#include "Image2D.h"
#include "ROI2D.h"
#include "Data2D.h"

namespace ncorr {

namespace details {
    class Strain2D_nlinfo_interpolator;
}
    
class Strain2D final { 
// ---------------------------------------------------------------------------//
// Strain2D is a class for 2D strains. ---------------------------------------//
// ---------------------------------------------------------------------------//
public:    
    typedef Data2D::difference_type                             difference_type;    
    typedef Data2D::coords                                               coords; 
    typedef details::Strain2D_nlinfo_interpolator           nlinfo_interpolator; 
    
    // Rule of 5 and destructor ----------------------------------------------//    
    Strain2D() noexcept = default;
    Strain2D(const Strain2D&) = default;
    Strain2D(Strain2D&&) noexcept = default;
    Strain2D& operator=(const Strain2D&) = default;
    Strain2D& operator=(Strain2D&&) = default; 
    ~Strain2D() noexcept = default;
    
    // Additional constructors -----------------------------------------------//
    Strain2D(Array2D<double> eyy, Array2D<double> exy, Array2D<double> exx, const ROI2D &roi, difference_type scalefactor) : // r-value
         eyy(std::move(eyy), roi, scalefactor), exy(std::move(exy), roi, scalefactor), exx(std::move(exx), roi, scalefactor) { }
        
    // Static factory methods ------------------------------------------------//
    static Strain2D load(std::ifstream&);

    // Operators interface ---------------------------------------------------//
    friend std::ostream& operator<<(std::ostream&, const Strain2D&); 
    friend void imshow(const Strain2D&, difference_type delay = -1);  
    friend bool isequal(const Strain2D&, const Strain2D&);
    friend void save(const Strain2D&, std::ofstream&);
        
    // Access ----------------------------------------------------------------//
    // Note that Strain2D is immutable, so all access is const.    
    difference_type data_height() const { return eyy.data_height(); }
    difference_type data_width() const { return eyy.data_width(); }
    const Data2D& get_eyy() const { return eyy; } 
    const Data2D& get_exy() const { return exy; } 
    const Data2D& get_exx() const { return exx; } 
    const ROI2D& get_roi() const { return eyy.get_roi(); }   
    difference_type get_scalefactor() const { return eyy.get_scalefactor(); }
    
    // Interpolator ----------------------------------------------------------//
    nlinfo_interpolator get_nlinfo_interpolator(difference_type, INTERP) const;    
                
    // Utility ---------------------------------------------------------------//
    std::string size_string() const { return eyy.size_string(); }   
    std::string size_2D_string() const { return eyy.size_2D_string(); }   
    
private:        
    Data2D eyy;   // Immutable - Data2D has pointer semantics
    Data2D exy;   // Immutable - Data2D has pointer semantics
    Data2D exx;   // Immutable - Data2D has pointer semantics
};
  
namespace details {    
    class Strain2D_nlinfo_interpolator final {    
        public:      
            typedef Strain2D::difference_type                   difference_type;   
            typedef Strain2D::coords                                     coords;
            
            friend Strain2D;
            
            // Rule of 5 and destructor --------------------------------------//
            Strain2D_nlinfo_interpolator() noexcept = default;
            Strain2D_nlinfo_interpolator(const Strain2D_nlinfo_interpolator&) = default;
            Strain2D_nlinfo_interpolator(Strain2D_nlinfo_interpolator&&) = default;
            Strain2D_nlinfo_interpolator& operator=(const Strain2D_nlinfo_interpolator&) = default;  
            Strain2D_nlinfo_interpolator& operator=(Strain2D_nlinfo_interpolator&&) = default;
            ~Strain2D_nlinfo_interpolator() noexcept = default;
            
            // Additional Constructors ---------------------------------------//            
            Strain2D_nlinfo_interpolator(const Strain2D &strain, difference_type region_idx, INTERP interp_type) :
                eyy_interp(strain.get_eyy().get_nlinfo_interpolator(region_idx,interp_type)), 
                exy_interp(strain.get_exy().get_nlinfo_interpolator(region_idx,interp_type)), 
                exx_interp(strain.get_exx().get_nlinfo_interpolator(region_idx,interp_type)) { }
             
            // Access methods ------------------------------------------------//
            std::tuple<double,double,double> operator()(double p1, double p2) const { return std::make_tuple(eyy_interp(p1,p2), exy_interp(p1,p2), exx_interp(p1,p2)); }
            std::tuple<const Array2D<double>&,const Array2D<double>&,const Array2D<double>&> first_order(double p1, double p2) const { return std::make_tuple(eyy_interp.first_order(p1,p2), exy_interp.first_order(p1,p2), exx_interp.first_order(p1,p2)); }
            
        private:       
            Data2D::nlinfo_interpolator eyy_interp; // must have copy of interpolator
            Data2D::nlinfo_interpolator exy_interp; // must have copy of interpolator
            Data2D::nlinfo_interpolator exx_interp; // must have copy of interpolator
    };    
}

}

#endif	/* STRAIN2D_H */

