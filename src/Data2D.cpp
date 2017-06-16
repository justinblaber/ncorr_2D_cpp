/* 
 * File:   Data2D.cpp
 * Author: justin
 * 
 * Created on February 28, 2015, 8:57 PM
 */

#include "Data2D.h"

namespace ncorr {

// Additional constructors ---------------------------------------------------//
Data2D::Data2D(Array2D<double> A, const ROI2D &roi, difference_type scalefactor) :
    scalefactor(scalefactor), roi(roi), A_ptr(std::make_shared<Array2D<double>>(std::move(A))) { 
    chk_scalefactor();
    chk_data_roi_same_size();
}    

Data2D::Data2D(Array2D<double> A, const ROI2D &roi) :
    scalefactor(1), roi(roi), A_ptr(std::make_shared<Array2D<double>>(std::move(A))) { 
    chk_data_roi_same_size();
}    

Data2D::Data2D(Array2D<double> A, difference_type scalefactor) :
    scalefactor(scalefactor), roi(Array2D<bool>(A.height(),A.width(),true)), A_ptr(std::make_shared<Array2D<double>>(std::move(A))) { 
    chk_scalefactor();
}    

Data2D::Data2D(Array2D<double> A) :
    scalefactor(1), roi(Array2D<bool>(A.height(),A.width(),true)), A_ptr(std::make_shared<Array2D<double>>(std::move(A))) { }

// Static factory methods ----------------------------------------------------//
Data2D Data2D::load(std::ifstream &is) {
    // Form empty Data2D then fill in values in accordance to how they are saved
    Data2D data;
    
    // Load scalefactor
    is.read(reinterpret_cast<char*>(&data.scalefactor), std::streamsize(sizeof(difference_type)));
    
    // Load roi
    data.roi = ROI2D::load(is);
    
    // Load A
    data.A_ptr = std::make_shared<Array2D<double>>(Array2D<double>::load(is));
    
    return data;
}

// Operators interface -------------------------------------------------------//
std::ostream& operator<<(std::ostream &os, const Data2D &data) {
    os << "Array: " << '\n' << *data.A_ptr;
    os << '\n' << "ROI: " << '\n' << data.roi;
    os << '\n' << "scale factor: " << data.scalefactor;
    
    return os;
}

void imshow(const Data2D &data, Data2D::difference_type delay) { 
    // Form buffer; set all values outside of ROI to slightly below minimum 
    // value of data, then show it, this guarantees the area outside the ROI is 
    // black.
    Array2D<double> A_buf = data.get_array();
    Array2D<double> A_data = A_buf(data.get_roi().get_mask());
    if (!A_data.empty()) {
        double A_min = min(A_data);
        // Subtract small amount so min value doesn't show as black.
        A_buf(~data.get_roi().get_mask()) = std::nextafter(A_min, A_min-1); 
    }    
    imshow(A_buf,delay);
}  

bool isequal(const Data2D &data1, const Data2D &data2) {
    return isequal(data1.get_array(), data2.get_array()) &&
           isequal(data1.get_roi(), data2.get_roi()) &&
           data1.scalefactor == data2.scalefactor;
}

void save(const Data2D &data, std::ofstream &os) {    
    typedef Data2D::difference_type                             difference_type;
    
    // Save scalefactor -> roi -> A
    os.write(reinterpret_cast<const char*>(&data.scalefactor), std::streamsize(sizeof(difference_type)));
    save(data.roi, os);
    save(*data.A_ptr, os);
}

// Interpolator --------------------------------------------------------------//
Data2D::nlinfo_interpolator Data2D::get_nlinfo_interpolator(difference_type region_idx, INTERP interp_type) const {
    return details::Data2D_nlinfo_interpolator(*this, region_idx, interp_type);
}

// Utility -------------------------------------------------------------------//
void Data2D::chk_scalefactor() const {
    if (scalefactor < 1) {
        throw std::invalid_argument("Attempted to form Data2D with scalefactor of: " + std::to_string(scalefactor) + 
                                    ". scalefactor must be an integer of value 1 or greater.");
    }
}

void Data2D::chk_data_roi_same_size() const {
    if (A_ptr->height() != roi.height() || A_ptr->width() != roi.width()) {
        throw std::invalid_argument("Attempted to form Data2D with Array2D of size: " + A_ptr->size_2D_string() + 
                                    " and ROI2D of size: " + roi.size_2D_string() + ". Sizes must be the same.");
    }
}

namespace details {
    struct sparse_tree_element final {
        typedef ROI2D::difference_type                          difference_type;
        
        // Constructor -------------------------------------------------------//
        sparse_tree_element(difference_type p1, difference_type p2, difference_type val) : p1(p1), p2(p2), val(val) { }
        
        // Arithmetic methods ------------------------------------------------//
        bool operator<(const sparse_tree_element &b) const { 
            if (p2 == b.p2) {
                return p1 < b.p1; // Sort by p1
            } else {
                return p2 < b.p2; // Sort by p2 first
            }
        };    
        
        difference_type p1;
        difference_type p2;
        mutable difference_type val;
    };
    
    void add_to_sparse_tree(std::set<sparse_tree_element> &sparse_tree, const sparse_tree_element &ste) {
        auto ste_it = sparse_tree.find(ste);
        if (ste_it == sparse_tree.end()) {
            // Val isnt in sparse_tree, so just insert it
            sparse_tree.insert(ste);
        } else {
            // Val is already in sparse_tree, so just modify the value
            ste_it->val += ste.val;
        }
    }
    
    Array2D<double>& inpaint_nlinfo(Array2D<double> &A, const ROI2D::region_nlinfo &nlinfo) {
        typedef ROI2D::difference_type                          difference_type;

        if (nlinfo.empty()) {
            // No digital inpainting if nlinfo is empty
            return A;
        }
        
        // Form mask ---------------------------------------------------------//
        Array2D<bool> mask_nlinfo(A.height(),A.width());
        fill(mask_nlinfo, nlinfo, true);    

        // Precompute inverse of nlinfo pixels' linear indices ---------------//
        Array2D<difference_type> A_inv_loc(A.height(),A.width(),-1); // -1 indicates pixel in nlinfo.
        difference_type inv_loc_counter = 0;
        for (difference_type p2 = 0; p2 < mask_nlinfo.width(); ++p2) {
            for (difference_type p1 = 0; p1 < mask_nlinfo.height(); ++p1) {
                if (!mask_nlinfo(p1,p2)) {
                    A_inv_loc(p1,p2) = inv_loc_counter++;
                }
            }
        }

        // Cycle over Array and form constraints -----------------------------//
        // Analyze points outside nlinfo AND boundary points
        std::set<sparse_tree_element> sparse_tree;
        std::vector<double> b;
        for (difference_type p2 = 0; p2 < A_inv_loc.width(); ++p2) {
            for (difference_type p1 = 0; p1 < A_inv_loc.height(); ++p1) {
                // Corners don't have constraints
                if ((p1 > 0 && p1 < A_inv_loc.height()-1) || (p2 > 0 && p2 < A_inv_loc.width()-1)) { 
                    // Sides have special constraints
                    if (p1 == 0 || p1 == A_inv_loc.height()-1) {
                        // Top or bottom
                        if (A_inv_loc(p1,p2-1) != -1 || A_inv_loc(p1,p2) != -1 || A_inv_loc(p1,p2+1) != -1) {
                            // Point of interest - add a constraint
                            b.push_back(0);
                            if (A_inv_loc(p1,p2-1) == -1) { b[b.size()-1] -= A(p1,p2-1); } else { add_to_sparse_tree(sparse_tree, {difference_type(b.size())-1, A_inv_loc(p1,p2-1), 1}); }
                            if (A_inv_loc(p1,p2) == -1)   { b[b.size()-1] += 2*A(p1,p2); } else { add_to_sparse_tree(sparse_tree, {difference_type(b.size())-1, A_inv_loc(p1,p2),  -2}); }
                            if (A_inv_loc(p1,p2+1) == -1) { b[b.size()-1] -= A(p1,p2+1); } else { add_to_sparse_tree(sparse_tree, {difference_type(b.size())-1, A_inv_loc(p1,p2+1), 1}); }
                        }
                    } else if (p2 == 0 || p2 == A_inv_loc.width()-1) {
                        // Left or right
                        if (A_inv_loc(p1-1,p2) != -1 || A_inv_loc(p1,p2) != -1 || A_inv_loc(p1+1,p2) != -1) {
                            // Point of interest - add a constraint
                            b.push_back(0);
                            if (A_inv_loc(p1-1,p2) == -1) { b[b.size()-1] -= A(p1-1,p2); } else { add_to_sparse_tree(sparse_tree, {difference_type(b.size())-1, A_inv_loc(p1-1,p2), 1}); }
                            if (A_inv_loc(p1,p2) == -1)   { b[b.size()-1] += 2*A(p1,p2); } else { add_to_sparse_tree(sparse_tree, {difference_type(b.size())-1, A_inv_loc(p1,p2),  -2}); }
                            if (A_inv_loc(p1+1,p2) == -1) { b[b.size()-1] -= A(p1+1,p2); } else { add_to_sparse_tree(sparse_tree, {difference_type(b.size())-1, A_inv_loc(p1+1,p2), 1}); }
                        }
                    } else {
                        // Center
                        if (A_inv_loc(p1-1,p2) != -1 || A_inv_loc(p1+1,p2) != -1 || A_inv_loc(p1,p2) != -1 || A_inv_loc(p1,p2-1) != -1 || A_inv_loc(p1,p2+1) != -1) {
                            // Point of interest - add a constraint
                            b.push_back(0);
                            if (A_inv_loc(p1-1,p2) == -1) { b[b.size()-1] -= A(p1-1,p2); } else { add_to_sparse_tree(sparse_tree, {difference_type(b.size())-1, A_inv_loc(p1-1,p2), 1}); }
                            if (A_inv_loc(p1+1,p2) == -1) { b[b.size()-1] -= A(p1+1,p2); } else { add_to_sparse_tree(sparse_tree, {difference_type(b.size())-1, A_inv_loc(p1+1,p2), 1}); }
                            if (A_inv_loc(p1,p2) == -1)   { b[b.size()-1] += 4*A(p1,p2); } else { add_to_sparse_tree(sparse_tree, {difference_type(b.size())-1, A_inv_loc(p1,p2),  -4}); }
                            if (A_inv_loc(p1,p2-1) == -1) { b[b.size()-1] -= A(p1,p2-1); } else { add_to_sparse_tree(sparse_tree, {difference_type(b.size())-1, A_inv_loc(p1,p2-1), 1}); }
                            if (A_inv_loc(p1,p2+1) == -1) { b[b.size()-1] -= A(p1,p2+1); } else { add_to_sparse_tree(sparse_tree, {difference_type(b.size())-1, A_inv_loc(p1,p2+1), 1}); }
                        }
                    }
                }                    
            }
        }    

        // Use sparse QR solver ----------------------------------------------//    
        // Note that later on maybe encapsulate free() into smart pointers. Make 
        // sure that cholmod_common is finished() last.
        // Start CHOLMOD
        cholmod_common c;
        cholmod_l_start(&c);   

        // Allocate and fill A
        cholmod_sparse *A_sparse = cholmod_l_allocate_sparse(b.size(),            // height
                                                             inv_loc_counter,     // width
                                                             sparse_tree.size(),  // # of elements
                                                             true,                // row indices are sorted
                                                             true,                // it is packed
                                                             0,                   // symmetry (0 = unsymmetric)
                                                             CHOLMOD_REAL,       
                                                             &c);    
        ((SuiteSparse_long*)A_sparse->p)[inv_loc_counter] = sparse_tree.size();   // Set last element before tree gets erased
        for (difference_type counter = 0, p2 = 0; p2 < inv_loc_counter; ++p2) {
            ((SuiteSparse_long*)A_sparse->p)[p2] = counter;
            while (!sparse_tree.empty() && p2 == sparse_tree.begin()->p2) {
                // Get first element
                auto it_ste = sparse_tree.begin(); 
                ((SuiteSparse_long*)A_sparse->i)[counter] = it_ste->p1;
                ((double*)A_sparse->x)[counter] = it_ste->val;
                sparse_tree.erase(it_ste); // delete element
                ++counter;
            } 
        }

        // Allocate and fill b
        cholmod_dense *b_dense = cholmod_l_allocate_dense(b.size(),       // Height
                                                          1,              // Width
                                                          b.size(),       // Leading dimension
                                                          CHOLMOD_REAL,
                                                          &c);  
        for (difference_type p = 0; p < difference_type(b.size()); ++p) {
            ((double*)b_dense->x)[p] = b[p];
        }

        // Solve and then fill results into inverse region of A
        // Note that documentation was hard to understand so I've done no error 
        // checking here (i.e. for out of memory) so maybe fix this later.
        cholmod_dense *x_dense = SuiteSparseQR<double>(A_sparse, b_dense, &c);        
        difference_type counter = 0;
        for (difference_type p2 = 0; p2 < mask_nlinfo.width(); ++p2) {
            for (difference_type p1 = 0; p1 < mask_nlinfo.height(); ++p1) {
                if (!mask_nlinfo(p1,p2)) {
                    A(p1,p2) = ((double*)x_dense->x)[counter++];
                }
            }
        }
                
        // Free memory
        cholmod_l_free_dense(&x_dense, &c);
        cholmod_l_free_dense(&b_dense, &c);
        cholmod_l_free_sparse(&A_sparse, &c);

        // Finish cholmod
        cholmod_l_finish(&c);        

        return A;
    }

    Data2D_nlinfo_interpolator::Data2D_nlinfo_interpolator(const Data2D &data, difference_type region_idx, INTERP interp_type) : 
        sub_data_ptr(std::make_shared<Array2D<double>>()), 
        sub_data_interp(sub_data_ptr->get_interpolator(interp_type)), 
        first_order_buf(3,1),
        scalefactor(data.get_scalefactor()), 
        nlinfo_top(), 
        nlinfo_left() { 
        // Perform digital inpainting of data contained in nlinfo first - this is
        // mainly for biquintic interpolation or any other form that requires an 
        // image patch for interpolation near edge points.
        
        // Get copy of nlinfo - copy needed because it is shift()'ed which alters it in-place.
        auto nlinfo = data.get_roi().get_nlinfo(region_idx);
        
        if (nlinfo.empty()) {
            // If this nlinfo is empty, there is nothing to interpolate. Note 
            // this will still return a functional interpolator that will return
            // NaNs.
            return;
        }
        
        // Store bounds - this allows conversion of interpolation coords 
        // (p1, p2) to local coordinates
        this->nlinfo_top = nlinfo.top;
        this->nlinfo_left = nlinfo.left;
        
        // Get sub_data array; use pinpainting so interpolation works nicely for 
        // boundary points.
        sub_data_ptr = std::make_shared<Array2D<double>>(nlinfo.bottom - nlinfo.top + 2*border + 1, 
                                                         nlinfo.right - nlinfo.left + 2*border + 1);
        (*sub_data_ptr)({border, sub_data_ptr->height() - border - 1},{border, sub_data_ptr->width() - border - 1}) = data.get_array()({nlinfo.top,nlinfo.bottom},{nlinfo.left,nlinfo.right});            
        inpaint_nlinfo(*sub_data_ptr, nlinfo.shift(border - nlinfo.top, border - nlinfo.left));
                        
        // Get interpolator
        sub_data_interp = sub_data_ptr->get_interpolator(interp_type);
    }
}

}