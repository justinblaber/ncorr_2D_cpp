/* 
 * File:   Disp2D.cpp
 * Author: justin
 * 
 * Created on May 12, 2015, 12:36 PM
 */

#include "Disp2D.h"

namespace ncorr {    
    
// Static factory methods ----------------------------------------------------//
Disp2D Disp2D::load(std::ifstream &is) {
    // Form empty Disp2D then fill in values in accordance to how they are saved
    Disp2D disp;
    
    // Load v
    disp.v = Data2D::load(is);
    
    // Load u
    disp.u = Data2D::load(is);
    
    return disp;
}
    
// Operators interface -------------------------------------------------------//
std::ostream& operator<<(std::ostream &os, const Disp2D &disp) {
    os << "V data: " << '\n' << disp.get_v();
    os << '\n' << "U data: " << '\n' << disp.get_u();
    
    return os;
}

void imshow(const Disp2D &disp, Disp2D::difference_type delay) { 
    // Just show each separately for now. If you combine into one buffer, you 
    // must scale each as their ranges might be different.
    imshow(disp.v, delay); 
    imshow(disp.u, delay); 
}  

bool isequal(const Disp2D &disp1, const Disp2D &disp2) {
    return isequal(disp1.v, disp2.v) && isequal(disp1.u, disp2.u);
}

void save(const Disp2D &disp, std::ofstream &os) {        
    // Save v -> u
    save(disp.v, os);
    save(disp.u, os);
}

// Interpolator --------------------------------------------------------------//
Disp2D::nlinfo_interpolator Disp2D::get_nlinfo_interpolator(difference_type region_idx, INTERP interp_type) const {
    return details::Disp2D_nlinfo_interpolator(*this, region_idx, interp_type);
}
    
}

