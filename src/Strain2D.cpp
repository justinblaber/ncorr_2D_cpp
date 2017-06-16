/* 
 * File:   Strain2D.cpp
 * Author: justin
 * 
 * Created on June 7, 2015, 10:14 PM
 */

#include "Strain2D.h"

namespace ncorr {    
    
// Static factory methods ----------------------------------------------------//
Strain2D Strain2D::load(std::ifstream &is) {
    // Form empty Strain2D then fill in values in accordance to how they are saved
    Strain2D strain;
    
    // Load eyy
    strain.eyy = Data2D::load(is);
    
    // Load exy
    strain.exy = Data2D::load(is);
    
    // Load exx
    strain.exx = Data2D::load(is);
    
    return strain;
}
    
// Operators interface -------------------------------------------------------//
std::ostream& operator<<(std::ostream &os, const Strain2D &strain) {
    os << "Eyy data: " << '\n' << strain.get_eyy();
    os << '\n' << "Exy data: " << '\n' << strain.get_exy();
    os << '\n' << "Exx data: " << '\n' << strain.get_exx();
    
    return os;
}

void imshow(const Strain2D &strain, Strain2D::difference_type delay) { 
    // Just show each separately for now. If you combine into one buffer, you 
    // must scale each as their ranges might be different.
    imshow(strain.eyy, delay); 
    imshow(strain.exy, delay); 
    imshow(strain.exx, delay); 
}  

bool isequal(const Strain2D &strain1, const Strain2D &strain2) {
    return isequal(strain1.eyy, strain2.eyy) && isequal(strain1.exy, strain2.exy) && isequal(strain1.exx, strain2.exx);
}

void save(const Strain2D &strain, std::ofstream &os) {        
    // Save eyy -> exy -> exx
    save(strain.eyy, os);
    save(strain.exy, os);
    save(strain.exx, os);
}

// Interpolator --------------------------------------------------------------//
Strain2D::nlinfo_interpolator Strain2D::get_nlinfo_interpolator(difference_type region_idx, INTERP interp_type) const {
    return details::Strain2D_nlinfo_interpolator(*this, region_idx, interp_type);
}
    
}
