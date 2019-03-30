/* 
 * File:   ncorr.cpp
 * Author: justin
 * 
 * Created on May 12, 2015, 1:33 AM
 */

#include "ncorr.h"

namespace ncorr {
    
namespace details {    
    // nonlinear optimizer ---------------------------------------------------//
    std::pair<const Array2D<double>&, bool> nloptimizer_base::global(const Array2D<double>& params_init) const {
        chk_input_params_size(params_init);
        
        // Copy initial params into params
        std::copy(params_init.get_pointer(), params_init.get_pointer() + params_init.size(), params.get_pointer());
        
        // Find initial guess
        bool success = initial_guess();
        
        // Performs iterative search only if initial guess is successful
        return success ? (*this)(params) : std::pair<const Array2D<double>&, bool>(params, false); 
    }
    
    std::pair<const Array2D<double>&, bool> nloptimizer_base::operator()(const Array2D<double>& params_guess) const {
        chk_input_params_size(params_guess);
        
        // Copy initial guess params into params
        std::copy(params_guess.get_pointer(), params_guess.get_pointer() + params_guess.size(), params.get_pointer());
                
        // Perform iterative search
        bool success = iterative_search();
                
        return { params, success };
    }    
    
    void nloptimizer_base::chk_input_params_size(const Array2D<double> &params) const {
        if (!params.same_size(this->params)) {
            throw std::invalid_argument("Attempted to input params of size: " + params.size_2D_string() + 
                                        " in nonlinear optimizer. Input params must have size of: " + this->params.size_2D_string() + ".");
        }
    }
    
    bool disp_nloptimizer::initial_guess() const {      
        // Note: params = {p1_new, p2_new, p1_old, p2_old, v_old, u_old, dv_dp1_old, dv_dp2_old, du_dp1_old, du_dp2_old, dist, grad_norm}
        
        // Get nlinfo corresponding to region_idx.
        const auto &nlinfo_old = disp.get_roi().get_nlinfo(region_idx);        
        if (nlinfo_old.empty()) {
            // If nlinfo_old is empty then an initial guess cannot be found.
            return false;
        }
        
        // Cycle over nlinfo_old and find integer coordinates where displacement 
        // maps closest to p1_new and p2_new, This will give the global estimate.
        double min_distance = std::numeric_limits<double>::max(); // Set to max so it gets updated
        for (difference_type nl_idx = 0; nl_idx < nlinfo_old.nodelist.width(); ++nl_idx) {
            difference_type p2_old_unscaled = nl_idx + nlinfo_old.left_nl;
            for (difference_type np_idx = 0; np_idx < nlinfo_old.noderange(nl_idx); np_idx += 2) {
                difference_type np_top_unscaled = nlinfo_old.nodelist(np_idx,nl_idx);
                difference_type np_bottom_unscaled = nlinfo_old.nodelist(np_idx + 1,nl_idx);
                for (difference_type p1_old_unscaled = np_top_unscaled; p1_old_unscaled <= np_bottom_unscaled; ++p1_old_unscaled) {   
                    // Get displacement directly from data array
                    double v_old = disp.get_v().get_array()(p1_old_unscaled, p2_old_unscaled);
                    double u_old = disp.get_u().get_array()(p1_old_unscaled, p2_old_unscaled);
                    
                    // Get old position
                    difference_type p1_old = p1_old_unscaled * disp.get_scalefactor();
                    difference_type p2_old = p2_old_unscaled * disp.get_scalefactor();
                    
                    // Calculate distance
                    double distance = std::sqrt(std::pow(params(0) - (p1_old + v_old), 2.0) + 
                                                std::pow(params(1) - (p2_old + u_old), 2.0));
                    
                    if (distance < min_distance) {
                        // Store params
                        params(2) = p1_old;
                        params(3) = p2_old;
                        params(4) = v_old;
                        params(5) = u_old;
                        min_distance = distance;
                    }                    
                }
            }
        }  
        // Initial guess will always succeed if nlinfo_old is nonempty
        return true; 
    }
    
    bool disp_nloptimizer::iterative_search() const {
        // Note: params = {p1_new, p2_new, p1_old, p2_old, v_old, u_old, dv_dp1_old, dv_dp2_old, du_dp1_old, du_dp2_old, dist, grad_norm}
                
        bool success = newton(); // Initial iteration - always perform at least 1
        for (difference_type counter = 1; success && params(11) > cutoff_norm && counter < cutoff_iterations; ++counter) { 
            success = newton();
        }        
        
        return success;
    }
    
    bool disp_nloptimizer::newton() const {
        // Note: params = {p1_new, p2_new, p1_old, p2_old, v_old, u_old, dv_dp1_old, dv_dp2_old, du_dp1_old, du_dp2_old, dist, grad_norm}
        
        // Interpolate value
        auto fo_pair = disp_interp.first_order(params(2), params(3));
        
        if (std::isnan(fo_pair.first(0))) {
            // Interpolated out of bounds; return as failure.
            return false;
        }
        
        // Get first order u and v references for simplicity
        const auto &fo_v = fo_pair.first;
        const auto &fo_u = fo_pair.second;
        
        // Determine gradient
        grad_buf(0) = 2 * (params(0) - params(2) - fo_v(0)) * (-1 - fo_v(1)) + 2 * (params(1) - params(3) - fo_u(0)) * (-fo_u(1));
        grad_buf(1) = 2 * (params(0) - params(2) - fo_v(0)) * (-fo_v(2)) + 2 * (params(1) - params(3) - fo_u(0)) * (-1 - fo_u(2));
     
        // Determine hessian
        hess_buf(0,0) = 2 * std::pow(-1 - fo_v(1), 2) + 2 * std::pow(-fo_u(1), 2);
        hess_buf(1,0) = 2 * -fo_v(2) * (-1 - fo_v(1)) + 2 * (-1 - fo_u(2)) * -fo_u(1);
        hess_buf(0,1) = hess_buf(1,0); // symmetric
        hess_buf(1,1) = 2 * std::pow(-fo_v(2), 2) + 2 * std::pow(-1 - fo_u(2), 2);        
                
        // Solve - hessian should be symmetric positive definite so use Cholesky
        // decomposition.
        auto hess_buf_linsolver = hess_buf.get_linsolver(LINSOLVER::CHOL);
        if (hess_buf_linsolver) { // Tests if hessian is actually positive definite
            // Update p1_old and p2_old
            const auto &delta_params = hess_buf_linsolver.solve(grad_buf);
                        
            params(2) -= delta_params(0);  // update p1_old
            params(3) -= delta_params(1);  // update p2_old
                        
            // Compute displacements/gradients at updated p1_old and p2_old positions
            auto fo_pair_updated = disp_interp.first_order(params(2), params(3));
            if (!std::isnan(fo_pair_updated.first(0))) {
                // Store values
                params(4) = fo_pair_updated.first(0);  // updated v
                params(5) = fo_pair_updated.second(0); // updated u
                params(6) = fo_pair_updated.first(1);  // updated dv/dp1
                params(7) = fo_pair_updated.first(2);  // updated dv/dp2
                params(8) = fo_pair_updated.second(1); // updated du/dp1
                params(9) = fo_pair_updated.second(2); // updated du/dp2
                params(11) = std::sqrt(dot(grad_buf,grad_buf)); // norm of gradient

                return true;
            }
        }
        // Something failed
        return false;
    }
    
    subregion_nloptimizer::subregion_nloptimizer(const Array2D<double> &A_ref, const Array2D<double> &A_cur, const ROI2D &roi, difference_type scalefactor, INTERP interp_type, SUBREGION subregion_type, difference_type r) : 
        nloptimizer_base(6, 10), 
        A_ref_ptr(std::make_shared<Array2D<double>>(A_ref)), 
        A_cur_ptr(std::make_shared<Array2D<double>>(A_cur)), 
        scalefactor(scalefactor), 
        A_cur_interp(A_cur.get_interpolator(interp_type)), 
        subregion_gen(roi.get_contig_subregion_generator(subregion_type, r)),
        A_cur_cumsum_p1_ptr(std::make_shared<Array2D<double>>(A_cur)), 
        A_cur_pow_cumsum_p1_ptr(std::make_shared<Array2D<double>>(pow(A_cur, 2.0))), 
        A_ref_template(2*r+1,2*r+1),      
        A_dref_dp1_ptr(std::make_shared<Array2D<double>>(A_ref.height(), A_ref.width())), 
        A_dref_dp2_ptr(std::make_shared<Array2D<double>>(A_ref.height(), A_ref.width())),
        ref_template_avg(),
        ref_template_ssd_inv(),      
        A_dref_dv(2*r+1,2*r+1),
        A_dref_du(2*r+1,2*r+1),
        A_dref_dv_dp1(2*r+1,2*r+1),
        A_dref_dv_dp2(2*r+1,2*r+1),
        A_dref_du_dp1(2*r+1,2*r+1),
        A_dref_du_dp2(2*r+1,2*r+1),
        A_cur_template(2*r+1,2*r+1) { 
        if (A_cur_ptr->height() < 2*r+1 || A_cur_ptr->width() < 2*r+1) {
            // A_cur must be larger than the ref template for cross correlation
            // function used in initial guess. 
            throw std::invalid_argument("Input current image to subregion optimizer has size of: " + A_cur_ptr->size_2D_string() + 
                                        ". Size must be bigger than twice the radius plus 1, which is: " + std::to_string(2*r+1) + ".");
        }
        
        // Precompute cumsum tables across p1 dimension
        for (difference_type p2 = 0; p2 < this->A_cur_ptr->width(); ++p2) {
            for (difference_type p1 = 1; p1 < this->A_cur_ptr->height(); ++p1) { // Skip first row
                (*this->A_cur_cumsum_p1_ptr)(p1,p2) += (*this->A_cur_cumsum_p1_ptr)(p1-1,p2);
                (*this->A_cur_pow_cumsum_p1_ptr)(p1,p2) += (*this->A_cur_pow_cumsum_p1_ptr)(p1-1,p2);
            }
        }
        // Precompute reference image gradients at integer locations
        auto A_ref_interp = A_ref.get_interpolator(interp_type);
        for (difference_type p2 = 0; p2 < this->A_ref_ptr->width(); ++p2) {
            for (difference_type p1 = 0; p1 < this->A_ref_ptr->height(); ++p1) {
                // Note that for some forms of interpolation, the entire array
                // cannot be interpolated (near boundaries - these values
                // will be NaNs). Possibly update this later.
                const auto &fo_ref = A_ref_interp.first_order(p1,p2);
                
                (*this->A_dref_dp1_ptr)(p1,p2) = fo_ref(1);
                (*this->A_dref_dp2_ptr)(p1,p2) = fo_ref(2);
            }
        }
    }
    
    bool subregion_nloptimizer::initial_guess() const {     
        // Perform (slightly modified) JP Lewis NCC for initial guess.        
        // Note: params = {p1, p2, v, u, dv_dp1, dv_dp2, du_dp1, du_dp2, corr_coef, diff_norm}
        
        // Get subregion nlinfo for input p1 and p2
        const auto &subregion_nlinfo = subregion_gen(std::round(params(0)), std::round(params(1)));
        
        // Form image patch for this subregion
        A_ref_template() = 0;
        ref_template_avg = 0.0;
        for (difference_type nl_idx = 0; nl_idx < subregion_nlinfo.nodelist.width(); ++nl_idx) {
            difference_type p2 = nl_idx + subregion_nlinfo.left_nl;
            for (difference_type np_idx = 0; np_idx < subregion_nlinfo.noderange(nl_idx); np_idx += 2) {
                difference_type np_top = subregion_nlinfo.nodelist(np_idx,nl_idx);
                difference_type np_bottom = subregion_nlinfo.nodelist(np_idx + 1,nl_idx);
                for (difference_type p1 = np_top; p1 <= np_bottom; ++p1) {   
                    difference_type p1_shifted = p1 - params(0) + subregion_gen.get_r();
                    difference_type p2_shifted = p2 - params(1) + subregion_gen.get_r();
                    
                    A_ref_template(p1_shifted, p2_shifted) = (*A_ref_ptr)(p1,p2);
                    ref_template_avg += A_ref_template(p1_shifted, p2_shifted);
                }
            }
        }        
        ref_template_avg /= subregion_nlinfo.points; // Finish
                
        // Calculate template_ssd_inv (ssd_inv = inverse sqrt of squared difference)
        // and subtract average from template. Note that A_ref_template will hold
        // the ref template with its average subtracted from it. Look at JP Lewis'
        // fast normalized cross correlation paper to see why.
        ref_template_ssd_inv = 0.0;
        for (difference_type nl_idx = 0; nl_idx < subregion_nlinfo.nodelist.width(); ++nl_idx) {
            difference_type p2 = nl_idx + subregion_nlinfo.left_nl;
            for (difference_type np_idx = 0; np_idx < subregion_nlinfo.noderange(nl_idx); np_idx += 2) {
                difference_type np_top = subregion_nlinfo.nodelist(np_idx,nl_idx);
                difference_type np_bottom = subregion_nlinfo.nodelist(np_idx + 1,nl_idx);
                for (difference_type p1 = np_top; p1 <= np_bottom; ++p1) {   
                    difference_type p1_shifted = p1 - params(0) + subregion_gen.get_r();
                    difference_type p2_shifted = p2 - params(1) + subregion_gen.get_r();
                    
                    A_ref_template(p1_shifted, p2_shifted) -= ref_template_avg;
                    ref_template_ssd_inv += std::pow(A_ref_template(p1_shifted, p2_shifted), 2.0);
                }
            }
        }
        ref_template_ssd_inv = std::sqrt(ref_template_ssd_inv);
        
        if (std::abs(ref_template_ssd_inv) >= std::numeric_limits<double>::epsilon()) {
            ref_template_ssd_inv = 1/ref_template_ssd_inv; // Finish
            
            // Must perform cross correlation of ref template with cur array
            auto A_xcorr = xcorr(*A_cur_ptr, A_ref_template);
                                   
            // Cycle and calculate maximum NCC - only cycle over non-overlapping
            // regions since xcorr uses a FFT and is circular.
            double NCC_max = -1.0; // range is [-1,1]
            difference_type p1_NCC_max = -1;
            difference_type p2_NCC_max = -1;
            for (difference_type p2 = params(1) - subregion_nlinfo.left; p2 < A_cur_ptr->width() - (subregion_nlinfo.right - params(1)); ++p2) {
                for (difference_type p1 = params(0) - subregion_nlinfo.top; p1 < A_cur_ptr->height() - (subregion_nlinfo.bottom - params(0)); ++p1) {
                    // Compute A_cur_sum and A_cur_pow_sum using cumsum tables
                    double A_cur_sum = 0;
                    double A_cur_pow_sum = 0;
                    for (difference_type nl_idx = 0; nl_idx < subregion_nlinfo.nodelist.width(); ++nl_idx) {
                        difference_type p2_shifted = nl_idx + subregion_nlinfo.left_nl - params(1) + p2;
                        for (difference_type np_idx = 0; np_idx < subregion_nlinfo.noderange(nl_idx); np_idx += 2) {
                            difference_type np_top_shifted = subregion_nlinfo.nodelist(np_idx,nl_idx) - params(0) + p1;
                            difference_type np_bottom_shifted = subregion_nlinfo.nodelist(np_idx + 1,nl_idx) - params(0) + p1;
                            
                            if (np_top_shifted > 0) {
                                A_cur_sum -= (*A_cur_cumsum_p1_ptr)(np_top_shifted-1, p2_shifted);
                                A_cur_pow_sum -= (*A_cur_pow_cumsum_p1_ptr)(np_top_shifted-1, p2_shifted);
                            }               
                            
                            A_cur_sum += (*A_cur_cumsum_p1_ptr)(np_bottom_shifted, p2_shifted);
                            A_cur_pow_sum += (*A_cur_pow_cumsum_p1_ptr)(np_bottom_shifted, p2_shifted);
                        }
                    }
                                        
                    double A_cur_ssd_inv = std::sqrt(A_cur_pow_sum - std::pow(A_cur_sum,2.0)/subregion_nlinfo.points);                    
                    if (std::abs(A_cur_ssd_inv) >= std::numeric_limits<double>::epsilon()) {
                        A_cur_ssd_inv = 1/A_cur_ssd_inv; // Finish
                        
                        // Calculate NCC value
                        double NCC_buf = A_xcorr(p1,p2) * ref_template_ssd_inv * A_cur_ssd_inv;
                        if (NCC_buf > NCC_max) {
                           NCC_max = NCC_buf; 
                           p1_NCC_max = p1;
                           p2_NCC_max = p2;
                        }
                    }
                }
            }           
            
            // Get displacements and return - note that there is guaranteed to
            // be at least one NCC point so no checking for max/min needs to be 
            // done.
            params(2) = p1_NCC_max - params(0); // v
            params(3) = p2_NCC_max - params(1); // u

            return true;
        }
        // Something failed
        return false;
    }
    
    bool subregion_nloptimizer::iterative_search() const {       
        // Note: params = {p1, p2, v, u, dv_dp1, dv_dp2, du_dp1, du_dp2, corr_coef, diff_norm}
        
        // Get nlinfo for input p1 and p2 - note that initial guess may have not
        // been called before iterative_search(), so recalculate subregion.
        const auto &subregion_nlinfo = subregion_gen(std::round(params(0)), std::round(params(1)));
        
        // Calculate average of ref template
        ref_template_avg = 0.0;
        for (difference_type nl_idx = 0; nl_idx < subregion_nlinfo.nodelist.width(); ++nl_idx) {
            difference_type p2 = nl_idx + subregion_nlinfo.left_nl;
            for (difference_type np_idx = 0; np_idx < subregion_nlinfo.noderange(nl_idx); np_idx += 2) {
                difference_type np_top = subregion_nlinfo.nodelist(np_idx,nl_idx);
                difference_type np_bottom = subregion_nlinfo.nodelist(np_idx + 1,nl_idx);
                for (difference_type p1 = np_top; p1 <= np_bottom; ++p1) {   
                    ref_template_avg += (*A_ref_ptr)(p1,p2);
                }
            }
        }        
        ref_template_avg /= subregion_nlinfo.points; // Finish
        
        // Calculate template_ssd_inv
        ref_template_ssd_inv = 0.0;
        for (difference_type nl_idx = 0; nl_idx < subregion_nlinfo.nodelist.width(); ++nl_idx) {
            difference_type p2 = nl_idx + subregion_nlinfo.left_nl;
            for (difference_type np_idx = 0; np_idx < subregion_nlinfo.noderange(nl_idx); np_idx += 2) {
                difference_type np_top = subregion_nlinfo.nodelist(np_idx,nl_idx);
                difference_type np_bottom = subregion_nlinfo.nodelist(np_idx + 1,nl_idx);
                for (difference_type p1 = np_top; p1 <= np_bottom; ++p1) {   
                    ref_template_ssd_inv += std::pow((*A_ref_ptr)(p1,p2) - ref_template_avg, 2.0);
                }
            }
        }
        ref_template_ssd_inv = std::sqrt(ref_template_ssd_inv);
        
        if (std::abs(ref_template_ssd_inv) >= std::numeric_limits<double>::epsilon()) {
            ref_template_ssd_inv = 1/ref_template_ssd_inv; // Finish
            
            // Precompute steepest descent images
            for (difference_type nl_idx = 0; nl_idx < subregion_nlinfo.nodelist.width(); ++nl_idx) {
                difference_type p2 = nl_idx + subregion_nlinfo.left_nl;
                for (difference_type np_idx = 0; np_idx < subregion_nlinfo.noderange(nl_idx); np_idx += 2) {
                    difference_type np_top = subregion_nlinfo.nodelist(np_idx,nl_idx);
                    difference_type np_bottom = subregion_nlinfo.nodelist(np_idx + 1,nl_idx);
                    for (difference_type p1 = np_top; p1 <= np_bottom; ++p1) {   
                        difference_type p1_shifted = p1 - params(0) + subregion_gen.get_r();
                        difference_type p2_shifted = p2 - params(1) + subregion_gen.get_r();
                        
                        double p1_delta = p1 - params(0);
                        double p2_delta = p2 - params(1);
                                                
                        if (std::isnan((*A_dref_dp1_ptr)(p1,p2))) {
                            // Note that for some forms of interpolation, the entire array
                            // cannot be interpolated (near boundaries - these values
                            // will be NaNs). Possibly update this later.
                            return false; 
                        }   
                        
                        A_dref_dv(p1_shifted,p2_shifted) = (*A_dref_dp1_ptr)(p1,p2);
                        A_dref_du(p1_shifted,p2_shifted) = (*A_dref_dp2_ptr)(p1,p2);
                        A_dref_dv_dp1(p1_shifted,p2_shifted) = (*A_dref_dp1_ptr)(p1,p2) * p1_delta;
                        A_dref_dv_dp2(p1_shifted,p2_shifted) = (*A_dref_dp1_ptr)(p1,p2) * p2_delta;
                        A_dref_du_dp1(p1_shifted,p2_shifted) = (*A_dref_dp2_ptr)(p1,p2) * p1_delta;
                        A_dref_du_dp2(p1_shifted,p2_shifted) = (*A_dref_dp2_ptr)(p1,p2) * p2_delta;
                    }
                }		
            }
            
            // Precompute GN Hessian
            hess_buf() = 0;
            for (difference_type nl_idx = 0; nl_idx < subregion_nlinfo.nodelist.width(); ++nl_idx) {
                difference_type p2 = nl_idx + subregion_nlinfo.left_nl;
                for (difference_type np_idx = 0; np_idx < subregion_nlinfo.noderange(nl_idx); np_idx += 2) {
                    difference_type np_top = subregion_nlinfo.nodelist(np_idx,nl_idx);
                    difference_type np_bottom = subregion_nlinfo.nodelist(np_idx + 1,nl_idx);
                    for (difference_type p1 = np_top; p1 <= np_bottom; ++p1) {  
                        difference_type p1_shifted = p1 - params(0) + subregion_gen.get_r();
                        difference_type p2_shifted = p2 - params(1) + subregion_gen.get_r();
                                               
                        // Hessian - only calculate lower half since hessian is symmetric
                        hess_buf(0,0) += A_dref_dv(p1_shifted,p2_shifted) * A_dref_dv(p1_shifted,p2_shifted);
                        hess_buf(1,0) += A_dref_dv(p1_shifted,p2_shifted) * A_dref_du(p1_shifted,p2_shifted);
                        hess_buf(2,0) += A_dref_dv(p1_shifted,p2_shifted) * A_dref_dv_dp1(p1_shifted,p2_shifted);
                        hess_buf(3,0) += A_dref_dv(p1_shifted,p2_shifted) * A_dref_dv_dp2(p1_shifted,p2_shifted);
                        hess_buf(4,0) += A_dref_dv(p1_shifted,p2_shifted) * A_dref_du_dp1(p1_shifted,p2_shifted);
                        hess_buf(5,0) += A_dref_dv(p1_shifted,p2_shifted) * A_dref_du_dp2(p1_shifted,p2_shifted);
                        
                        hess_buf(1,1) += A_dref_du(p1_shifted,p2_shifted) * A_dref_du(p1_shifted,p2_shifted);
                        hess_buf(2,1) += A_dref_du(p1_shifted,p2_shifted) * A_dref_dv_dp1(p1_shifted,p2_shifted);
                        hess_buf(3,1) += A_dref_du(p1_shifted,p2_shifted) * A_dref_dv_dp2(p1_shifted,p2_shifted);
                        hess_buf(4,1) += A_dref_du(p1_shifted,p2_shifted) * A_dref_du_dp1(p1_shifted,p2_shifted);
                        hess_buf(5,1) += A_dref_du(p1_shifted,p2_shifted) * A_dref_du_dp2(p1_shifted,p2_shifted);
                        
                        hess_buf(2,2) += A_dref_dv_dp1(p1_shifted,p2_shifted) * A_dref_dv_dp1(p1_shifted,p2_shifted);
                        hess_buf(3,2) += A_dref_dv_dp1(p1_shifted,p2_shifted) * A_dref_dv_dp2(p1_shifted,p2_shifted);
                        hess_buf(4,2) += A_dref_dv_dp1(p1_shifted,p2_shifted) * A_dref_du_dp1(p1_shifted,p2_shifted);
                        hess_buf(5,2) += A_dref_dv_dp1(p1_shifted,p2_shifted) * A_dref_du_dp2(p1_shifted,p2_shifted);
                        
                        hess_buf(3,3) += A_dref_dv_dp2(p1_shifted,p2_shifted) * A_dref_dv_dp2(p1_shifted,p2_shifted);
                        hess_buf(4,3) += A_dref_dv_dp2(p1_shifted,p2_shifted) * A_dref_du_dp1(p1_shifted,p2_shifted);
                        hess_buf(5,3) += A_dref_dv_dp2(p1_shifted,p2_shifted) * A_dref_du_dp2(p1_shifted,p2_shifted);
                        
                        hess_buf(4,4) += A_dref_du_dp1(p1_shifted,p2_shifted) * A_dref_du_dp1(p1_shifted,p2_shifted);
                        hess_buf(5,4) += A_dref_du_dp1(p1_shifted,p2_shifted) * A_dref_du_dp2(p1_shifted,p2_shifted);
                        
                        hess_buf(5,5) += A_dref_du_dp2(p1_shifted,p2_shifted) * A_dref_du_dp2(p1_shifted,p2_shifted);
                    }
                }
            }
            // Finish hessian
            for (difference_type p2 = 0; p2 < hess_buf.width(); p2++) {
                for (difference_type p1 = p2; p1 < hess_buf.height(); p1++) {
                    hess_buf(p1,p2) *= 2 * std::pow(ref_template_ssd_inv, 2.0); // Finish
                    hess_buf(p2,p1) = hess_buf(p1,p2);                          // Fill upperhalf
                }
            }
                        
            // Get linsolver - use Cholesky decomposition since hessian should 
            // be symmetric positive definite.
            hess_linsolver = hess_buf.get_linsolver(LINSOLVER::CHOL);            
            if (hess_linsolver) {
                // Note: params = {p1, p2, v, u, dv_dp1, dv_dp2, du_dp1, du_dp2, corr_coef, diff_norm}                                
                bool success = newton(); // Initial iteration - always perform at least 1
                for (difference_type counter = 1; success && params(9) > cutoff_norm && counter < cutoff_iterations; ++counter) { 
                    success = newton();
                }                      
                
                return success;
            }
        }        
        // Something failed
        return false;
    }
        
    bool subregion_nloptimizer::newton() const {   
        // Note: params = {p1, p2, v, u, dv_dp1, dv_dp2, du_dp1, du_dp2, corr_coef, diff_norm}
        
        // subregion_nlinfo is already guaranteed to be set since 
        // iterative_search() is always called before newton(), so just get a 
        // reference to the subregion from subregion_gen.
        const auto &subregion_nlinfo = subregion_gen.get_subregion_nlinfo();
        
        A_cur_template() = 0;
        double cur_template_avg = 0.0;
        for (difference_type nl_idx = 0; nl_idx < subregion_nlinfo.nodelist.width(); ++nl_idx) {
            difference_type p2 = nl_idx + subregion_nlinfo.left_nl;
            for (difference_type np_idx = 0; np_idx < subregion_nlinfo.noderange(nl_idx); np_idx += 2) {
                difference_type np_top = subregion_nlinfo.nodelist(np_idx,nl_idx);
                difference_type np_bottom = subregion_nlinfo.nodelist(np_idx + 1,nl_idx);
                for (difference_type p1 = np_top; p1 <= np_bottom; ++p1) {   
                    difference_type p1_shifted = p1 - params(0) + subregion_gen.get_r();
                    difference_type p2_shifted = p2 - params(1) + subregion_gen.get_r();

                    double p1_delta = p1 - params(0);
                    double p2_delta = p2 - params(1);
                    
                    // Find transformed coordinates
                    double p1_transformed = p1 + params(2) + params(4) * p1_delta + params(5) * p2_delta;
                    double p2_transformed = p2 + params(3) + params(6) * p1_delta + params(7) * p2_delta;
                    
                    // Interpolate and store value
                    A_cur_template(p1_shifted, p2_shifted) = A_cur_interp(p1_transformed, p2_transformed);
                    
                    if (std::isnan(A_cur_template(p1_shifted, p2_shifted))) {
                        // If interpolated out of range, return false.
                        return false; 
                    }               
                    
                    cur_template_avg += A_cur_template(p1_shifted, p2_shifted);
                }
            }
        }        
        cur_template_avg /= subregion_nlinfo.points; // Finish
                
        // Calculate cur_template_ssd_inv
        double cur_template_ssd_inv = 0.0;
        for (difference_type nl_idx = 0; nl_idx < subregion_nlinfo.nodelist.width(); ++nl_idx) {
            difference_type p2 = nl_idx + subregion_nlinfo.left_nl;
            for (difference_type np_idx = 0; np_idx < subregion_nlinfo.noderange(nl_idx); np_idx += 2) {
                difference_type np_top = subregion_nlinfo.nodelist(np_idx,nl_idx);
                difference_type np_bottom = subregion_nlinfo.nodelist(np_idx + 1,nl_idx);
                for (difference_type p1 = np_top; p1 <= np_bottom; ++p1) {   
                    difference_type p1_shifted = p1 - params(0) + subregion_gen.get_r();
                    difference_type p2_shifted = p2 - params(1) + subregion_gen.get_r();
                    
                    cur_template_ssd_inv += std::pow(A_cur_template(p1_shifted,p2_shifted) - cur_template_avg, 2.0);
                }
            }
        }
        cur_template_ssd_inv = std::sqrt(cur_template_ssd_inv);
                
        if (std::abs(cur_template_ssd_inv) >= std::numeric_limits<double>::epsilon()) {
            cur_template_ssd_inv = 1/cur_template_ssd_inv; // Finish
        
            // Calculate gradient and correlation coefficient
            grad_buf() = 0.0;
            params(8) = 0.0; // correlation coefficient
            for (difference_type nl_idx = 0; nl_idx < subregion_nlinfo.nodelist.width(); ++nl_idx) {
                difference_type p2 = nl_idx + subregion_nlinfo.left_nl;
                for (difference_type np_idx = 0; np_idx < subregion_nlinfo.noderange(nl_idx); np_idx += 2) {
                    difference_type np_top = subregion_nlinfo.nodelist(np_idx,nl_idx);
                    difference_type np_bottom = subregion_nlinfo.nodelist(np_idx + 1,nl_idx);
                    for (difference_type p1 = np_top; p1 <= np_bottom; ++p1) {   
                        difference_type p1_shifted = p1 - params(0) + subregion_gen.get_r();
                        difference_type p2_shifted = p2 - params(1) + subregion_gen.get_r();
                        
                        double normalized_diff = ((*A_ref_ptr)(p1,p2) - ref_template_avg) * ref_template_ssd_inv - (A_cur_template(p1_shifted,p2_shifted) - cur_template_avg) * cur_template_ssd_inv;
                        
                        grad_buf(0) += normalized_diff * A_dref_dv(p1_shifted,p2_shifted);
                        grad_buf(1) += normalized_diff * A_dref_du(p1_shifted,p2_shifted);
                        grad_buf(2) += normalized_diff * A_dref_dv_dp1(p1_shifted,p2_shifted);
                        grad_buf(3) += normalized_diff * A_dref_dv_dp2(p1_shifted,p2_shifted);
                        grad_buf(4) += normalized_diff * A_dref_du_dp1(p1_shifted,p2_shifted);
                        grad_buf(5) += normalized_diff * A_dref_du_dp2(p1_shifted,p2_shifted);
                        
                        // correlation coefficient
                        params(8) += std::pow(normalized_diff, 2.0);
                    }
                }
            }
                    
            // Finish gradient (note: make this negative since delta_p is solved
            // with the negative gradient)
            for (difference_type p = 0; p < grad_buf.height(); ++p) {
                grad_buf(p) *= -2 * ref_template_ssd_inv;
            }
                        
            // Find delta_p
            const auto &delta_p = hess_linsolver.solve(grad_buf);
                        
            // Calculate norm of delta_p
            params(9) = std::sqrt(dot(delta_p,delta_p));
            
            // Update parameters using inverse composition            
            double denominator = (delta_p(5) + delta_p(2) + delta_p(5)*delta_p(2) - delta_p(4)*delta_p(3) + 1);
            if (std::abs(denominator) >= std::numeric_limits<double>::epsilon()) {
                double v = params(2);
                double u = params(3);
                double dv_dp1 = params(4);
                double dv_dp2 = params(5);
                double du_dp1 = params(6);
                double du_dp2 = params(7);
            
                params(2) = v - ((dv_dp1+1)*(delta_p(0)*(1+delta_p(5))-delta_p(1)*delta_p(3)) + dv_dp2*(delta_p(1)*(1+delta_p(2))-delta_p(0)*delta_p(4)))/denominator;
                params(3) = u - ((du_dp2+1)*(delta_p(1)*(1+delta_p(2))-delta_p(0)*delta_p(4)) + du_dp1*(delta_p(0)*(1+delta_p(5))-delta_p(1)*delta_p(3)))/denominator;
                params(4) = ((delta_p(5)+1)*(dv_dp1+1) - delta_p(4)*dv_dp2)/denominator - 1; 
                params(5) = (dv_dp2*(delta_p(2)+1) - delta_p(3)*(dv_dp1+1))/denominator; 
                params(6) = (du_dp1*(delta_p(5)+1) - delta_p(4)*(du_dp2+1))/denominator; 
                params(7) = ((delta_p(2)+1)*(du_dp2+1) - delta_p(3)*du_dp1)/denominator - 1;
                
                return true;
            }
        }
        // Something failed
        return false;
    }
}

// Interface functions -------------------------------------------------------//
namespace details {
    Array2D<double> update_boundary(const Array2D<double> &boundary, const Disp2D::nlinfo_interpolator &disp_interp, const ROI2D::difference_type roi_scalefactor) {
        typedef ROI2D::difference_type                          difference_type;
        
        // For now, points going out of data bounds are left as-is. Preferably,
        // I'd perform a polygon intersection of the boundary with the data size,
        // but polygon intersection is very complex to perform in double 
        // precision.
        
        auto boundary_updated = boundary; // Make a copy
        for (difference_type idx = 0; idx < boundary_updated.height(); ++idx) {
            // Get displacement at position of boundary
            auto disp_pair = disp_interp(boundary_updated(idx,0) * roi_scalefactor, boundary_updated(idx,1) * roi_scalefactor);
                        
            if (std::isnan(disp_pair.first)) {
                // Boundary interpolated out of range - return an empty boundary
                // to signal failure.
                return Array2D<double>(0,2);
            }
            
            // Add displacement to boundary position.
            boundary_updated(idx,0) += disp_pair.first / roi_scalefactor;
            boundary_updated(idx,1) += disp_pair.second / roi_scalefactor;
        }    
        
        return boundary_updated;
    }
}

ROI2D update(const ROI2D &roi, const Disp2D &disp, INTERP interp_type) {
    typedef ROI2D::difference_type                              difference_type;
    
    // regions in 'roi' and 'disp' must correspond
    if (roi.size_regions() != disp.get_roi().size_regions()) {
        throw std::invalid_argument("Attempted to update ROI2D which has " + std::to_string(roi.size_regions()) + 
                                    " regions with a Disp2D which has " + std::to_string(disp.get_roi().size_regions()) + 
                                    " regions. The number of regions must be the same and they must correspond to each other.");
    }
    
    // Update only supported for:
    //  1) Reduced ROI2D with the same data size as the Disp2D. In this case,
    //     the scalefactor of the ROI2D is assumed to be the same as Disp2D's 
    //     scalefactor.
    //  2) Full sized ROI2D with a reduced size (based on Disp2D's scalefactor) 
    //     the same as Disp2D. In this case, the scalefactor of the ROI2D is 
    //     assumed to be 1.
    difference_type roi_scalefactor;
    if (roi.height() == disp.data_height() && roi.width() == disp.data_width()) {
        roi_scalefactor = disp.get_scalefactor();
    } else if (std::ceil(double(roi.height()) / disp.get_scalefactor()) == disp.data_height() &&
               std::ceil(double(roi.width()) / disp.get_scalefactor()) == disp.data_width()) {
        roi_scalefactor = 1;
    } else {
        throw std::invalid_argument("Attempted to update ROI2D which has size of: " + roi.size_2D_string() + 
                                    " with a Disp2D which has data size of: " + disp.size_2D_string() + " and a scale factor of: " + std::to_string(disp.get_scalefactor()) +
                                    ". Size of ROI2D or reduced ROI2D must match data size of Disp2D.");
    }
        
    // Update roi by updating each region boundary
    std::vector<ROI2D::region_boundary> boundaries_updated(roi.size_regions());
    for (difference_type region_idx = 0; region_idx < roi.size_regions(); ++region_idx) {
        if (disp.get_roi().get_nlinfo(region_idx).empty()) {
            // If nlinfo in disp is empty for this region, continue since no 
            // interpolation can be done.
            continue;
        }
        
        // Get nlinfo interpolator from disp
        auto disp_interp = disp.get_nlinfo_interpolator(region_idx, interp_type);
        
        // Get region_boundary
        const auto &boundary = roi.get_boundary(region_idx); 
        
        // Form new boundary -------------------------------------------------//
        ROI2D::region_boundary boundary_updated;  
        
        // Update "add" boundary first    
        boundary_updated.add = details::update_boundary(boundary.add, disp_interp, roi_scalefactor);        
        
        // Update "sub" boundaries next
        for (auto &sub_boundary : boundary.sub) {
            boundary_updated.sub.emplace_back(details::update_boundary(sub_boundary, disp_interp, roi_scalefactor));
        } 
        
        // Store boundary
        boundaries_updated[region_idx] = std::move(boundary_updated);
    }    
    
    return ROI2D(std::move(boundaries_updated), roi.height(), roi.width());
}

namespace details {          
    Array2D<ROI2D::difference_type>& get_nlinfo_SDA(Array2D<ROI2D::difference_type> &SDA, const ROI2D::region_nlinfo &nlinfo, Array2D<bool> &mask_buf, Array2D<bool> &mask_buf_new) {
        typedef ROI2D::difference_type                          difference_type;
            
        if (nlinfo.empty()) {
            // For empty nlinfo nothing is changed
            return SDA;    
        }
        
        // Computes signed distance array for an nlinfo. Note that this is 
        // probably not the fastest way to this, but it is simple/easy/correct
        // and calculating the SDA is typically not a bottleneck.
        
        mask_buf() = true;
        fill(mask_buf, nlinfo, false);
        mask_buf_new() = false;
        difference_type dist = 1; // Initialize to 1 since SDA will be 0 outside ROI
        bool points_added = true; // Keeps track if points were added to the border
        while (points_added) { 
            // While loop will expand border by 1 pixel per iteration and exit 
            // when the ROI is filled.
            points_added = false;
            for (difference_type nl_idx = 0; nl_idx < nlinfo.nodelist.width(); ++nl_idx) {
                difference_type p2 = nl_idx + nlinfo.left_nl;
                for (difference_type np_idx = 0; np_idx < nlinfo.noderange(nl_idx); np_idx += 2) {
                    difference_type np_top = nlinfo.nodelist(np_idx,nl_idx);
                    difference_type np_bottom = nlinfo.nodelist(np_idx + 1,nl_idx);
                    for (difference_type p1 = np_top; p1 <= np_bottom; ++p1) {
                        if (!mask_buf(p1,p2)) {
                            // This point has not been analyzed yet. See if 
                            // any of the four neighboring points is true.
                            if ((mask_buf.in_bounds(p1 - 1,p2) && mask_buf(p1 - 1, p2)) || // Up
                                (mask_buf.in_bounds(p1 + 1,p2) && mask_buf(p1 + 1, p2)) || // Down 
                                (mask_buf.in_bounds(p1,p2 - 1) && mask_buf(p1, p2 - 1)) || // Left
                                (mask_buf.in_bounds(p1,p2 + 1) && mask_buf(p1, p2 + 1))) { // Right
                                // This is a boundary point, so set its value and
                                // set mask_buf_new to mark that it has been 
                                // analyzed.
                                SDA(p1,p2) = dist;
                                mask_buf_new(p1,p2) = true;
                                points_added = true;
                            }      
                        }				
                    }
                }
            }
            
            // Transfer nlinfo of mask_buf_new to mask_buf
            for (difference_type nl_idx = 0; nl_idx < nlinfo.nodelist.width(); ++nl_idx) {
                difference_type p2 = nl_idx + nlinfo.left_nl;
                for (difference_type np_idx = 0; np_idx < nlinfo.noderange(nl_idx); np_idx += 2) {
                    difference_type np_top = nlinfo.nodelist(np_idx,nl_idx);
                    difference_type np_bottom = nlinfo.nodelist(np_idx + 1,nl_idx);
                    for (difference_type p1 = np_top; p1 <= np_bottom; ++p1) {
                        mask_buf(p1,p2) = mask_buf_new(p1,p2);
                    }
                }
            }          
            
            // Increment dist
            ++dist;
        }
        
        return SDA;
    }
    
    Array2D<ROI2D::difference_type> get_ROI_SDA(const ROI2D &roi) {
        typedef ROI2D::difference_type                          difference_type;
        
        // Initialize SDA and two mask buffers - be sure to initialize SDA to 0
        Array2D<difference_type> SDA(roi.height(), roi.width()); 
        Array2D<bool> mask_buf1(roi.height(), roi.width());
        Array2D<bool> mask_buf2(roi.height(), roi.width());
        for (difference_type region_idx = 0; region_idx < roi.size_regions(); ++region_idx) {
            // Form SDA for each region            
            if (!roi.get_nlinfo(region_idx).empty()) {
                get_nlinfo_SDA(SDA, roi.get_nlinfo(region_idx), mask_buf1, mask_buf2);
            }
        }

        return SDA;
    }
    
    Array2D<double> get_seed_params(const Data2D &data, 
                                    const ROI2D::region_nlinfo &nlinfo, 
                                    const disp_nloptimizer &d_nloptimizer, 
                                    const Array2D<ROI2D::difference_type> &SDA,
                                    Array2D<double> &params_buf) {        
        // Note: params = {p1_new, p2_new, p1_old, p2_old, v_old, u_old, dv_dp1_old, dv_dp2_old, du_dp1_old, du_dp2_old, dist, grad_norm}
        
        if (nlinfo.empty()) {
            // If nlinfo is empty, cannot get seed params. Return empty Array2D
            // to signal failure
            return Array2D<double>();
        }
        
        // Find maximum SDA location inside this region - use this as initial seed point
        auto max_info = max(SDA, nlinfo);
        
        // Set params for nonlinear optimizer
        params_buf(0) = max_info.second.first * data.get_scalefactor();
        params_buf(1) = max_info.second.second * data.get_scalefactor();
        params_buf(10) = max_info.first;

        // Get rest of params using global search
        auto seed_params_pair = d_nloptimizer.global(params_buf);      
        if (seed_params_pair.second) {
            // Successful 
            return seed_params_pair.first;
        }
        // Something failed
        return Array2D<double>();
    }    
    
    bool analyze_point(const Array2D<double> &queue_params,
                       ROI2D::difference_type p1_new_delta, 
                       ROI2D::difference_type p2_new_delta, 
                       const ROI2D::region_nlinfo &nlinfo,             
                       ROI2D::difference_type roi_scalefactor, 
                       const disp_nloptimizer &d_nloptimizer,
                       const Array2D<ROI2D::difference_type> &SDA,
                       std::priority_queue<Array2D<double>, std::vector<Array2D<double>>, std::function<bool(const Array2D<double>&,const Array2D<double>&)>> &queue, 
                       Array2D<bool> &A_new_ap, 
                       Array2D<double> &params_buf) {
        typedef ROI2D::difference_type                          difference_type; 
        
        // Note: params = {p1_new, p2_new, p1_old, p2_old, v_old, u_old, dv_dp1_old, dv_dp2_old, du_dp1_old, du_dp2_old, dist, grad_norm}
        
        // Make sure points are in bounds of roi_new's nlinfo and haven't been 
        // analyzed yet.
        difference_type p1_new = queue_params(0) + p1_new_delta;
        difference_type p2_new = queue_params(1) + p2_new_delta;
        if (nlinfo.in_nlinfo(p1_new / roi_scalefactor, p2_new / roi_scalefactor) &&
            !A_new_ap(p1_new  / roi_scalefactor, p2_new / roi_scalefactor)) {
            A_new_ap(p1_new / roi_scalefactor, p2_new / roi_scalefactor) = true; // Mark as analyzed
            
            // Make sure denominator isn't close to zero
            double denominator = (queue_params(8) * queue_params(7) - (1 + queue_params(6)) * (1 + queue_params(9)));            
            if (std::abs(denominator) >= std::numeric_limits<double>::epsilon()) {
                // Get initial guess using gradients            
                double p1_old_delta = (queue_params(7) * (p2_new - (queue_params(3) + queue_params(5))) - (1 + queue_params(9)) * (p1_new - (queue_params(2) + queue_params(4)))) / denominator;
                double p2_old_delta = (queue_params(8) * (p1_new - (queue_params(2) + queue_params(4))) - (1 + queue_params(6)) * (p2_new - (queue_params(3) + queue_params(5)))) / denominator;

                // Fill in params
                params_buf(0) = p1_new;
                params_buf(1) = p2_new;
                params_buf(2) = queue_params(2) + p1_old_delta;
                params_buf(3) = queue_params(3) + p2_old_delta; 
                params_buf(10) = SDA(p1_new  / roi_scalefactor, p2_new / roi_scalefactor); 

                // Get optimized params from nonlinear optimizer
                auto params_pair = d_nloptimizer(params_buf);                
                if (params_pair.second) {
                    // Insert into queue
                    queue.push(params_pair.first);
                    
                    return true;
                } else {
                    // Optimization failed using guess, so try global optimization.
                    auto params_global_pair = d_nloptimizer.global(params_buf);
                    if (params_global_pair.second) {
                        // Insert into queue
                        queue.push(params_global_pair.first);
                        
                        return true;
                    }
                }
            }
        }
        // Something failed
        return false;
    }
}

Data2D update(const Data2D &data, const Disp2D &disp, INTERP interp_type) {
    typedef ROI2D::difference_type                              difference_type;
    
    // regions in 'data' and 'disp' must correspond
    if (data.get_roi().size_regions() != disp.get_roi().size_regions()) {
        throw std::invalid_argument("Attempted to update Data2D which has " + std::to_string(data.get_roi().size_regions()) + 
                                    " regions with a Disp2D which has " + std::to_string(disp.get_roi().size_regions()) + 
                                    " regions. The number of regions must be the same and they must correspond to each other.");
    }
    
    // Update only supported for:
    //  1) Reduced Data2D with the same data size as the Disp2D and equal
    //     scalefactors.
    //  2) Full sized Data2D with a reduced size the same Disp2D
    if (!(data.get_scalefactor() == disp.get_scalefactor() && data.data_height() == disp.data_height() && data.data_width() == disp.data_width()) &&
        !(data.get_scalefactor() == 1 && std::ceil(double(data.data_height()) / disp.get_scalefactor()) == disp.data_height() && std::ceil(double(data.data_width()) / disp.get_scalefactor()) == disp.data_width())) {
        throw std::invalid_argument("Attempted to update Data2D which has a data size of " + data.size_2D_string() + " and scale factor of: " + std::to_string(data.get_scalefactor()) +
                                    " with a Disp2D which has a data size of " + disp.size_2D_string() + " and scale factor of: " + std::to_string(disp.get_scalefactor()) +
                                    ". Data2D must either have the same scalefactor as Disp2D with the same data size, or Data2D must be full sized with a reduced size the same as Disp2D.");
    }
    
    // Update ROI and then fill in updated regions.
    auto roi_new = update(data.get_roi(), disp, interp_type);
    
    // Get signed distance array for roi_new - this guides queue such that 
    // interior points are updated first.
    auto SDA = details::get_ROI_SDA(roi_new);
    
    // Form A_new and other buffers and then fill in values in updated ROI
    Array2D<double> A_new(data.data_height(), data.data_width());  // Newly interpolated array
    Array2D<bool> A_new_ap(data.data_height(), data.data_width()); // Analyzed points
    Array2D<bool> A_new_vp(data.data_height(), data.data_width()); // Valid points
    Array2D<double> params_buf(12, 1);
    for (difference_type region_idx = 0; region_idx < roi_new.size_regions(); ++region_idx) {     
        if (disp.get_roi().get_nlinfo(region_idx).empty() || roi_new.get_nlinfo(region_idx).empty()) {
            // If either nlinfo is empty, there cannot be any updates for this region
            continue;
        }
        
        // Get nonlinear displacement optimizer for this region
        details::disp_nloptimizer d_nloptimizer(disp, region_idx, interp_type);
        
        // Get data nlinfo interpolator
        auto data_interp = data.get_nlinfo_interpolator(region_idx, interp_type);
                               
        // Get seed params for queue
        auto seed_params = details::get_seed_params(data, roi_new.get_nlinfo(region_idx), d_nloptimizer, SDA, params_buf);
        if (!seed_params.empty()) {                        
            A_new_ap(seed_params(0) / data.get_scalefactor(), seed_params(1) / data.get_scalefactor()) = true; // Mark as analyzed
                   
            // Form queue - make it a priority queue based on SDA value. This
            // will analyze points of high distance from boundary first.
            auto comp = [](const Array2D<double> &a, const Array2D<double> &b ) { return a(10) < b(10); };
            std::priority_queue<Array2D<double>, std::vector<Array2D<double>>, std::function<bool(const Array2D<double>&,const Array2D<double>&)>> queue(comp);
            
            // Perform flood fill around seed  
            queue.push(seed_params);                            
            while (!queue.empty()) {			
                // 1) pop info from queue                
                auto queue_params = std::move(queue.top()); queue.pop();
                
                // 2) Validate and interpolate point
                A_new(queue_params(0) / data.get_scalefactor(),  queue_params(1) / data.get_scalefactor()) = data_interp(queue_params(2),  queue_params(3));
                if (!std::isnan(A_new(queue_params(0) / data.get_scalefactor(),  queue_params(1) / data.get_scalefactor()))) {
                    A_new_vp(queue_params(0) / data.get_scalefactor(),  queue_params(1) / data.get_scalefactor()) = true;
                }
                                           
                // 3) Analyze four surrounding points - up, down, left right;
                details::analyze_point(queue_params, -data.get_scalefactor(), 0, roi_new.get_nlinfo(region_idx), data.get_scalefactor(), d_nloptimizer, SDA, queue, A_new_ap, params_buf);
                details::analyze_point(queue_params,  data.get_scalefactor(), 0, roi_new.get_nlinfo(region_idx), data.get_scalefactor(), d_nloptimizer, SDA, queue, A_new_ap, params_buf);
                details::analyze_point(queue_params, 0, -data.get_scalefactor(), roi_new.get_nlinfo(region_idx), data.get_scalefactor(), d_nloptimizer, SDA, queue, A_new_ap, params_buf);
                details::analyze_point(queue_params, 0,  data.get_scalefactor(), roi_new.get_nlinfo(region_idx), data.get_scalefactor(), d_nloptimizer, SDA, queue, A_new_ap, params_buf);
            }    
        }
    }
    // Must form union with valid points    
    return Data2D(std::move(A_new), roi_new.form_union(A_new_vp), data.get_scalefactor());
}

Disp2D add(const std::vector<Disp2D> &disps, INTERP interp_type) {
    typedef ROI2D::difference_type                              difference_type;
    
    // This will add displacements WRT the configuration of the first displacement
    // plot.
    
    if (disps.empty()) {
        // Return empty disp
        return Disp2D();
    } else if (disps.size() == 1) {
        // Return input disp
        return disps.front();
    }
    
    // Adding displacements only supported for displacements which have the same
    // scalefactor, data size, and number of regions.
    difference_type scalefactor = disps.front().get_scalefactor();
    const auto &roi = disps.front().get_roi();
    for (difference_type disp_idx = 1; disp_idx < difference_type(disps.size()); ++disp_idx) {
        if (disps[disp_idx].get_scalefactor() != scalefactor ||
            disps[disp_idx].data_height() != disps.front().data_height() ||
            disps[disp_idx].data_width() != disps.front().data_width() ||
            disps[disp_idx].get_roi().size_regions() != roi.size_regions()) {
            throw std::invalid_argument("Attempted to add displacements with differing scalefactors, sizes, or number of regions. All scalefactors, sizes, and number of regions must be the same.");
        }
    }
    
    // Form buffers and add one region at a time
    Array2D<bool> A_vp(roi.height(), roi.width());       
    Array2D<double> A_v_added(roi.height(), roi.width());
    Array2D<double> A_u_added(roi.height(), roi.width());
    for (difference_type region_idx = 0; region_idx < roi.size_regions(); ++region_idx) {
        // Get nlinfo interpolators for each displacement field for this region
        std::vector<Disp2D::nlinfo_interpolator> disp_nlinfo_interps;
        for (difference_type disp_idx = 0; disp_idx < difference_type(disps.size()); ++disp_idx) {
            disp_nlinfo_interps.push_back(disps[disp_idx].get_nlinfo_interpolator(region_idx, interp_type));
        }
        
        // Add displacements for this region
        for (difference_type nl_idx = 0; nl_idx < roi.get_nlinfo(region_idx).nodelist.width(); ++nl_idx) {
            difference_type p2_unscaled = nl_idx + roi.get_nlinfo(region_idx).left_nl;
            difference_type p2 = p2_unscaled * scalefactor;
            for (difference_type np_idx = 0; np_idx < roi.get_nlinfo(region_idx).noderange(nl_idx); np_idx += 2) {
                difference_type np_top = roi.get_nlinfo(region_idx).nodelist(np_idx,nl_idx);
                difference_type np_bottom = roi.get_nlinfo(region_idx).nodelist(np_idx + 1,nl_idx);
                for (difference_type p1_unscaled = np_top; p1_unscaled <= np_bottom; ++p1_unscaled) {
                    difference_type p1 = p1_unscaled * scalefactor;
                    
                    // Cycle over displacements and add displacement values
                    double v_added = 0;
                    double u_added = 0;
                    for (difference_type disp_idx = 0; disp_idx < difference_type(disps.size()); ++disp_idx) {
                        // Make sure coords are close to this nlinfo - the 
                        // further away points are from the roi, the more 
                        // extrapolation is required and hence less accurate.
                        bool near_nlinfo = false;
                        difference_type window = 1;
                        difference_type p1_unscaled = std::round((p1 + v_added) / scalefactor);
                        difference_type p2_unscaled = std::round((p2 + u_added) / scalefactor);
                        for (difference_type p2_window = p2_unscaled - window; p2_window <= p2_unscaled + window; ++p2_window) {
                            for (difference_type p1_window = p1_unscaled - window; p1_window <= p1_unscaled + window; ++p1_window) {
                                if (disps[disp_idx].get_roi().get_nlinfo(region_idx).in_nlinfo(p1_window,p2_window)) {
                                    near_nlinfo = true;
                                }
                                if (near_nlinfo) { break; }
                            }
                            if (near_nlinfo) { break; }
                        }
                        
                        if (near_nlinfo) {
                            auto disp_pair = disp_nlinfo_interps[disp_idx](p1 + v_added, p2 + u_added);

                            v_added += disp_pair.first;
                            u_added += disp_pair.second;
                        } else {
                            // Set to NaN if position is not within ROI - this
                            // will trigger a break
                            v_added = std::numeric_limits<double>::quiet_NaN();
                            u_added = std::numeric_limits<double>::quiet_NaN();
                        }
                                                
                        if (std::isnan(v_added)) {
                            // This interpolated out of bounds, so break
                            break;
                        }
                    }
                    
                    if (!std::isnan(v_added)) {
                        // Only add point if it was interpolated in bounds
                        A_v_added(p1_unscaled, p2_unscaled) = v_added;
                        A_u_added(p1_unscaled, p2_unscaled) = u_added;
                        A_vp(p1_unscaled, p2_unscaled) = true;
                    }
                }
            }
        }
    }
    // Must form union with valid points
    return { std::move(A_v_added), std::move(A_u_added), roi.form_union(A_vp), scalefactor };  
}

// DIC_analysis --------------------------------------------------------------//
namespace details {        
    Array2D<double> get_centroid(const ROI2D::region_nlinfo &nlinfo) {
        typedef ROI2D::difference_type                          difference_type;
        
        if (nlinfo.empty()) {
            // Will return NaN if nlinfo is empty.
            return Array2D<double>({ std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN() });
        }
        
        double p1_centroid = 0;
        double p2_centroid = 0;        
        for (difference_type nl_idx = 0; nl_idx < nlinfo.nodelist.width(); ++nl_idx) {
            difference_type p2 = nl_idx + nlinfo.left_nl;
            for (difference_type np_idx = 0; np_idx < nlinfo.noderange(nl_idx); np_idx += 2) {
                difference_type np_top = nlinfo.nodelist(np_idx,nl_idx);
                difference_type np_bottom = nlinfo.nodelist(np_idx + 1,nl_idx);
                for (difference_type p1 = np_top; p1 <= np_bottom; ++p1) {
                    p1_centroid += p1;
                    p2_centroid += p2;                    
                }
            }
        }
        // Finish centroids
        p1_centroid /= nlinfo.points;
        p2_centroid /= nlinfo.points;
        
        return Array2D<double>({ p1_centroid, p2_centroid });
    }
    
    std::pair<std::vector<ROI2D::region_nlinfo>,std::vector<ROI2D::region_nlinfo>> nlinfo_line_split(const Array2D<double> &direc,
                                                                                                     const Array2D<double> &origin,
                                                                                                     ROI2D::difference_type partition_offset,
                                                                                                     const ROI2D::region_nlinfo &nlinfo,
                                                                                                     Array2D<ROI2D::difference_type> &partition_diagram,
                                                                                                     Array2D<bool> &mask_buf) {
        typedef ROI2D::difference_type                          difference_type;
        
        // Divides regions across input vector at specified origin - note that 
        // this split may not be contiguous.
        
        for (difference_type nl_idx = 0; nl_idx < nlinfo.nodelist.width(); ++nl_idx) {
            difference_type p2 = nl_idx + nlinfo.left_nl;
            for (difference_type np_idx = 0; np_idx < nlinfo.noderange(nl_idx); np_idx += 2) {
                difference_type np_top = nlinfo.nodelist(np_idx,nl_idx);
                difference_type np_bottom = nlinfo.nodelist(np_idx + 1,nl_idx);
                for (difference_type p1 = np_top; p1 <= np_bottom; ++p1) {
                    // See if (p1,p2) is on the left or right side of the input
                    // vector.             
                    double point1_p1 = origin(0),            point1_p2 = origin(1);                
                    double point2_p1 = origin(0) + direc(0), point2_p2 = origin(1) + direc(1);               
                    double point3_p1 = p1,                   point3_p2 = p2; 
                    
                    // Note that this "turn" predicate is simple and not robust, 
                    // but the values used here will generally be well behaved
                    // so this shouldn't be an issue.        
                    double turn_val = point2_p2*point3_p1 - point2_p1*point3_p2 - point1_p2*(point3_p1-point2_p1) + point1_p1*(point3_p2-point2_p2);
                    if (std::abs(turn_val) < std::numeric_limits<double>::epsilon()) {
                        // Lies on the axis
                        partition_diagram(p1,p2) = partition_offset;
                    } else if (turn_val > 0){
                        // Right side of axis
                        partition_diagram(p1,p2) = partition_offset;
                    } else {                        
                        // Left side of axis
                        partition_diagram(p1,p2) = partition_offset+1;
                    }
                    
                }
            }
        }
        
        // Return split nlinfos
        mask_buf() = false;
        fill(mask_buf, nlinfo, true);
        mask_buf = (std::move(mask_buf) & (partition_diagram == partition_offset));
        auto nlinfo_part1_pair = ROI2D::region_nlinfo::form_nlinfos(mask_buf);
        
        mask_buf() = false;
        fill(mask_buf, nlinfo, true);
        mask_buf = (std::move(mask_buf) & (partition_diagram == partition_offset+1));
        auto nlinfo_part2_pair = ROI2D::region_nlinfo::form_nlinfos(mask_buf);
                
        return { std::move(nlinfo_part1_pair.first), std::move(nlinfo_part2_pair.first) };
    }
    
    void nlinfo_contig_expansion(const ROI2D::region_nlinfo &sub_nlinfo,
                                 const ROI2D::region_nlinfo &nlinfo,
                                 ROI2D::difference_type val, 
                                 Array2D<ROI2D::difference_type> &A,
                                 Array2D<bool> &A_ap) {
        typedef ROI2D::difference_type                          difference_type;
                
        // Performs contiguous expansion for the sub_nlinfo which is within 
        // nlinfo.
        
        std::queue<difference_type> queue;    
        for (difference_type nl_idx = 0; nl_idx < sub_nlinfo.nodelist.width(); ++nl_idx) {
            difference_type p2 = nl_idx + sub_nlinfo.left_nl;
            for (difference_type np_idx = 0; np_idx < sub_nlinfo.noderange(nl_idx); np_idx += 2) {
                difference_type np_top = sub_nlinfo.nodelist(np_idx,nl_idx);
                difference_type np_bottom = sub_nlinfo.nodelist(np_idx + 1,nl_idx);
                for (difference_type p1 = np_top; p1 <= np_bottom; ++p1) {
                    if (nlinfo.in_nlinfo(p1-1,p2) && A_ap(p1-1,p2)) { A_ap(p1-1,p2) = false; queue.push(p1-1); queue.push(p2); }
                    if (nlinfo.in_nlinfo(p1+1,p2) && A_ap(p1+1,p2)) { A_ap(p1+1,p2) = false; queue.push(p1+1); queue.push(p2); }
                    if (nlinfo.in_nlinfo(p1,p2+1) && A_ap(p1,p2+1)) { A_ap(p1,p2+1) = false; queue.push(p1); queue.push(p2+1); }
                    if (nlinfo.in_nlinfo(p1,p2-1) && A_ap(p1,p2-1)) { A_ap(p1,p2-1) = false; queue.push(p1); queue.push(p2-1); }
                }
            }
        }
        
        while (!queue.empty()) {
            difference_type p1 = queue.front(); queue.pop();
            difference_type p2 = queue.front(); queue.pop();

            // Fill plot
            A(p1,p2) = val;

            // Check four surrounding neighbors
            if (nlinfo.in_nlinfo(p1-1,p2) && A_ap(p1-1,p2)) { A_ap(p1-1,p2) = false; queue.push(p1-1); queue.push(p2); }
            if (nlinfo.in_nlinfo(p1+1,p2) && A_ap(p1+1,p2)) { A_ap(p1+1,p2) = false; queue.push(p1+1); queue.push(p2); }
            if (nlinfo.in_nlinfo(p1,p2+1) && A_ap(p1,p2+1)) { A_ap(p1,p2+1) = false; queue.push(p1); queue.push(p2+1); }
            if (nlinfo.in_nlinfo(p1,p2-1) && A_ap(p1,p2-1)) { A_ap(p1,p2-1) = false; queue.push(p1); queue.push(p2-1); }
        }
    }
            
    std::pair<std::pair<ROI2D::region_nlinfo,ROI2D::region_nlinfo>, bool> nlinfo_contig_split(ROI2D::difference_type partition_offset, 
                                                                                              const ROI2D::region_nlinfo &nlinfo,
                                                                                              Array2D<ROI2D::difference_type> &partition_diagram,
                                                                                              Array2D<bool> &mask_buf) {
        typedef ROI2D::difference_type                          difference_type;
        
        // This will attempt to split the nlinfo contiguously by finding the 
        // centroid, major axis, and minor axis and then partitioning the region
        // base on these values. If successful, the two split regions are 
        // returned as nlinfos.
        
        if (nlinfo.empty()) {
            // Cannot split an empty nlinfo, so return as failed.
            return { { ROI2D::region_nlinfo(), ROI2D::region_nlinfo() } , false };
        }
        
        // Find centroid
        auto centroid = get_centroid(nlinfo);
        
        // Find moments
        double Mp1p1 = 0;  
        double Mp1p2 = 0;  
        double Mp2p2 = 0;      
        for (difference_type nl_idx = 0; nl_idx < nlinfo.nodelist.width(); ++nl_idx) {
            difference_type p2 = nl_idx + nlinfo.left_nl;
            for (difference_type np_idx = 0; np_idx < nlinfo.noderange(nl_idx); np_idx += 2) {
                difference_type np_top = nlinfo.nodelist(np_idx,nl_idx);
                difference_type np_bottom = nlinfo.nodelist(np_idx + 1,nl_idx);
                for (difference_type p1 = np_top; p1 <= np_bottom; ++p1) {
                    Mp1p1 += std::pow(p1 - centroid(0),2);
                    Mp1p2 += (p1 - centroid(0)) * (p2 - centroid(1));
                    Mp2p2 += std::pow(p2 - centroid(1),2);
                }
            }
        }
        // Finish moments
        Mp1p1 /= nlinfo.points;
        Mp1p2 /= nlinfo.points;
        Mp2p2 /= nlinfo.points;
        
        // Obtain eigenvectors and eigenvalues - since this is a simple 2x2 
        // matrix, a direct analytic solution can be used.
        double eig_val1 = 0.0;
        double eig_val2 = 0.0;
        Array2D<double> eig_vec1(2,1);
        Array2D<double> eig_vec2(2,1);
        if (std::abs(Mp1p2) >= std::numeric_limits<double>::epsilon()) {
            double trace = Mp1p1 + Mp2p2;
            double det = Mp1p1*Mp2p2 - std::pow(Mp1p2,2);
        
            eig_val1 = trace/2.0 + std::sqrt((std::pow(trace,2.0)/4.0 - det));
            eig_val2 = trace/2.0 - std::sqrt((std::pow(trace,2.0)/4.0 - det));
        
            eig_vec1(0,0) = eig_val1-Mp2p2; eig_vec1(1,0) = Mp1p2;
            eig_vec2(0,0) = eig_val2-Mp2p2; eig_vec2(1,0) = Mp1p2;
            
            // Normalize eigenvectors
            eig_vec1 = normalize(std::move(eig_vec1));
            eig_vec2 = normalize(std::move(eig_vec2));
        } else {
            // Edge condition - matrix is diagonal
            eig_val1 = Mp1p1;
            eig_val2 = Mp2p2;
            eig_vec1(0,0) = 1; eig_vec1(1,0) = 0;
            eig_vec2(0,0) = 0; eig_vec2(1,0) = 1;
        }
        
        // Get major and minor axes
        const auto &minor_axis = eig_val1 <= eig_val2 ? eig_vec1 : eig_vec2;
        const auto &major_axis = eig_val1 >  eig_val2 ? eig_vec1 : eig_vec2;
        
        // 1) Attempt to split across minor axis first.
        // 2) If this results in more than two regions, then attempt to split
        //    across the major axis
        // 3) If this results in more than two regions, then use the split that
        //    results in the two largest regions and then contiguously grow the 
        //    regions (starting with the smaller one first) until the entire 
        //    nlinfo is split in two.
                
        // 1) Try minor axis split
        auto nlinfos_minor_split_pair = nlinfo_line_split(minor_axis, centroid, partition_offset, nlinfo, partition_diagram, mask_buf);         
        if (nlinfos_minor_split_pair.first.size() == 0 || nlinfos_minor_split_pair.second.size() == 0) {
            // Region is too small to be split, so return as failed
            return { { ROI2D::region_nlinfo(), ROI2D::region_nlinfo() } , false };
        }
        
        if (nlinfos_minor_split_pair.first.size() != 1 || nlinfos_minor_split_pair.second.size() != 1) {
            // 2) Try major axis split next
            auto nlinfos_major_split_pair = nlinfo_line_split(major_axis, centroid, partition_offset, nlinfo, partition_diagram, mask_buf);    
            if (nlinfos_major_split_pair.first.size() == 0 || nlinfos_major_split_pair.second.size() == 0) {
                // Region is too small to be split, so return as failed.
                return { { ROI2D::region_nlinfo(), ROI2D::region_nlinfo() } , false };
            }
            
            if (nlinfos_major_split_pair.first.size() != 1 || nlinfos_major_split_pair.second.size() != 1) {
                // 3) See which split has the two largest contiguous regions and
                //    then perform contiguous expansion.
                auto nlinfo_compare = [](const ROI2D::region_nlinfo &a, const ROI2D::region_nlinfo &b) { return a.points < b.points; };
                
                const auto &minor_largest1 = *std::max_element(nlinfos_minor_split_pair.first.begin(),  nlinfos_minor_split_pair.first.end(),  nlinfo_compare);
                const auto &minor_largest2 = *std::max_element(nlinfos_minor_split_pair.second.begin(), nlinfos_minor_split_pair.second.end(), nlinfo_compare);
                
                const auto &major_largest1 = *std::max_element(nlinfos_major_split_pair.first.begin(),  nlinfos_major_split_pair.first.end(),  nlinfo_compare);
                const auto &major_largest2 = *std::max_element(nlinfos_major_split_pair.second.begin(), nlinfos_major_split_pair.second.end(), nlinfo_compare);
                
                difference_type minor_sum = minor_largest1.points + minor_largest2.points;                
                difference_type major_sum = major_largest1.points + major_largest2.points;
                
                // Must make contiguous again --------------------------------//
                // Get the nlinfos used for contiguous expansion
                const auto &nlinfo_larger  = minor_sum > major_sum ? (minor_largest2.points >= minor_largest1.points ? minor_largest2 : minor_largest1) : 
                                                                     (major_largest2.points >= major_largest1.points ? major_largest2 : major_largest1);
                const auto &nlinfo_smaller = minor_sum > major_sum ? (minor_largest2.points <  minor_largest1.points ? minor_largest2 : minor_largest1) : 
                                                                     (major_largest2.points <  major_largest1.points ? major_largest2 : major_largest1);
                               
                auto partition_num_larger  = minor_sum > major_sum ? (minor_largest2.points >= minor_largest1.points ? partition_offset+1 : partition_offset) : 
                                                                     (major_largest2.points >= major_largest1.points ? partition_offset+1 : partition_offset);
                auto partition_num_smaller = minor_sum > major_sum ? (minor_largest2.points <  minor_largest1.points ? partition_offset+1 : partition_offset) : 
                                                                     (major_largest2.points <  major_largest1.points ? partition_offset+1 : partition_offset);
                
                // Use mask buf to keep track of which points are left that need 
                // to be analyzed (like A_ap).
                mask_buf() = false;
                fill(mask_buf, nlinfo, true);
                fill(mask_buf, nlinfo_larger,  false);
                fill(mask_buf, nlinfo_smaller, false);
                
                // Refill partition diagram
                fill(partition_diagram, nlinfo_larger,  partition_num_larger);
                fill(partition_diagram, nlinfo_smaller, partition_num_smaller);
                
                // Perform a contiguous expansion for each region, starting with
                // the smaller nlinfo first
                nlinfo_contig_expansion(nlinfo_smaller, nlinfo, partition_num_smaller, partition_diagram, mask_buf);
                nlinfo_contig_expansion(nlinfo_larger,  nlinfo, partition_num_larger,  partition_diagram, mask_buf);  
            }
        }
                
        // Get nlinfo for first partition
        mask_buf() = false;
        fill(mask_buf, nlinfo, true);
        mask_buf = (std::move(mask_buf) & (partition_diagram == partition_offset));
        auto nlinfo_part1_pair = ROI2D::region_nlinfo::form_nlinfos(mask_buf);

        // Get nlinfo for second partition
        mask_buf() = false;
        fill(mask_buf, nlinfo, true);
        mask_buf = (std::move(mask_buf) & (partition_diagram == partition_offset+1));
        auto nlinfo_part2_pair = ROI2D::region_nlinfo::form_nlinfos(mask_buf);
        
        if (nlinfo_part1_pair.first.size() != 1 || nlinfo_part2_pair.first.size() != 1) {
            // This is a programmer error
            throw std::invalid_argument("nlinfo_contig_split() did not actually form 2 contiguous subregions");
        }
        
        return { { std::move(nlinfo_part1_pair.first.front()), std::move(nlinfo_part2_pair.first.front()) }, true };
    }
        
    bool recursive_nlinfo_partition_diagram(ROI2D::difference_type part_num1, 
                                            ROI2D::difference_type part_num2,
                                            const ROI2D::region_nlinfo &nlinfo, 
                                            Array2D<ROI2D::difference_type> &partition_diagram,
                                            Array2D<bool> &mask_buf) {
        typedef ROI2D::difference_type                          difference_type;
        
        // Exit condition
        if (part_num2 == part_num1) {
            // Finish partition and then return
            fill(partition_diagram, nlinfo, part_num1);
            
            return true;            
        } else {
            // Partition again
            auto nlinfo_split_pair = nlinfo_contig_split(part_num1, nlinfo, partition_diagram, mask_buf);
                                         
            if (nlinfo_split_pair.second) {
                // Region was split successfully, continue to recurse.
                difference_type part_num_middle = (part_num1 + part_num2)/2;                
                return recursive_nlinfo_partition_diagram(part_num1,   part_num_middle, nlinfo_split_pair.first.first,  partition_diagram, mask_buf) &&
                       recursive_nlinfo_partition_diagram(part_num_middle+1, part_num2, nlinfo_split_pair.first.second, partition_diagram, mask_buf);   
            } 
        }
        // Something failed
        return false;   
    }
        
    Array2D<ROI2D::difference_type>& get_nlinfo_partition_diagram(const ROI2D::region_nlinfo &nlinfo, 
                                                                  ROI2D::difference_type num_partitions,
                                                                  Array2D<ROI2D::difference_type> &partition_diagram,
                                                                  Array2D<bool> &mask_buf) {
        
        // Perform divide and conquer algorithm to partition this region.
        bool success = recursive_nlinfo_partition_diagram(0, 
                                                          num_partitions-1, 
                                                          nlinfo,
                                                          partition_diagram,
                                                          mask_buf);
        
        if (!success) {
            // Fill this nlinfo back to -1 if it was not successfully partitioned
            fill(partition_diagram, nlinfo, -1);
        }
        
        return partition_diagram;
    }
        
    Array2D<ROI2D::difference_type> get_ROI_partition_diagram(const ROI2D &roi, ROI2D::difference_type num_partitions) {
        typedef ROI2D::difference_type                          difference_type;
        
        // If ROI is partitioned successfully, partitioned regions will contain
        // numbers from 0 to num_partitions-1. Outside of ROI, partition diagram 
        // will be -1. If partitioning for a region fails (if too many 
        // partitions are requested and region is small), that region will 
        // contain -1s.
        
        // Initialize partition_diagram and buffers
        Array2D<difference_type> partition_diagram(roi.height(), roi.width(), -1); // Initialize to -1
        Array2D<bool> mask_buf(roi.height(), roi.width());                         
        for (difference_type region_idx = 0; region_idx < roi.size_regions(); ++region_idx) {         
            if (!roi.get_nlinfo(region_idx).empty()) {
                get_nlinfo_partition_diagram(roi.get_nlinfo(region_idx), num_partitions, partition_diagram, mask_buf);
            }
        }
        
        return partition_diagram;
    }
    
    Array2D<ROI2D::difference_type> get_nlinfo_partition_diagram_seeds(const Array2D<ROI2D::difference_type> &partition_diagram, 
                                                                       const ROI2D::region_nlinfo &nlinfo, 
                                                                       ROI2D::difference_type num_partitions,
                                                                       Array2D<ROI2D::difference_type> &SDA,
                                                                       Array2D<bool> &mask_buf1,
                                                                       Array2D<bool> &mask_buf2) {
        typedef ROI2D::difference_type                          difference_type;
        
        // Cycle over each partition number and get its seed
        Array2D<difference_type> seed_buf(num_partitions,2);
        for (difference_type partition_idx = 0; partition_idx < num_partitions; ++partition_idx) {
            // Form nlinfo corresponding to this partition within input nlinfo
            mask_buf1() = false;
            fill(mask_buf1, nlinfo, true);
            mask_buf1 = (std::move(mask_buf1) & (partition_diagram == partition_idx));            
            auto nlinfos_pair = ROI2D::region_nlinfo::form_nlinfos(mask_buf1);
                
            // Test to make sure there is only one 4-way contiguous region
            if (nlinfos_pair.first.size() == 0) {
                // This nlinfo was not partitioned (or is empty), return empty seeds
                return Array2D<difference_type>(0,2);
            } else if (nlinfos_pair.first.size() != 1) {
                // This is a programmer error.
                throw std::invalid_argument("While calculating partition seeds, it was determined that one of the partitioned regions was not 4-way contiguous.");
            }

            // Get SDA corresponding to the partition within this nlinfo
            get_nlinfo_SDA(SDA, nlinfos_pair.first.front(), mask_buf1, mask_buf2);

            // Find maximum SDA location - use this as the seed point. Note that
            // this nlinfo is non-empty so max is guaranteed to exist.
            auto max_info = max(SDA, nlinfos_pair.first.front());

            // Store max location
            seed_buf(partition_idx,0) = max_info.second.first;
            seed_buf(partition_idx,1) = max_info.second.second;
        }

        return seed_buf;
    }
    
    std::vector<Array2D<ROI2D::difference_type>> get_ROI_partition_diagram_seeds(const Array2D<ROI2D::difference_type> &partition_diagram, const ROI2D &roi, ROI2D::difference_type num_partitions) {
        typedef ROI2D::difference_type                          difference_type;

        // This will obtain seed positions for each region's partitions by 
        // finding the most interior point of each partition.
        
        // Cycle over regions and get seeds for each partition
        std::vector<Array2D<difference_type>> partition_seeds(roi.size_regions());   
        Array2D<difference_type> SDA(roi.height(), roi.width()); // Make sure to initialize to 0
        Array2D<bool> mask_buf1(roi.height(), roi.width()); 
        Array2D<bool> mask_buf2(roi.height(), roi.width());
        for (difference_type region_idx = 0; region_idx < roi.size_regions(); ++region_idx) {
            if (!roi.get_nlinfo(region_idx).empty()) {
                partition_seeds[region_idx] = get_nlinfo_partition_diagram_seeds(partition_diagram, roi.get_nlinfo(region_idx), num_partitions, SDA, mask_buf1, mask_buf2);
            }
        }
        
        return partition_seeds;        
    } 
        
    Array2D<double> get_seed_params(const Array2D<ROI2D::difference_type> &seeds_pos,
                                    const subregion_nloptimizer &sr_nloptimizer,
                                    Array2D<double> &params_buf) {
        typedef ROI2D::difference_type                          difference_type; 
               
        // Note: params = {p1, p2, v, u, dv_dp1, dv_dp2, du_dp1, du_dp2, corr_coef, diff_norm}
                
        // Cycle over seeds and return the seed which has the lowest correlation 
        // coefficient
        
        Array2D<double> seed_params;
        for (difference_type seed_idx = 0; seed_idx < seeds_pos.height(); ++seed_idx) {
            params_buf(0) = seeds_pos(seed_idx,0);
            params_buf(1) = seeds_pos(seed_idx,1);
            
            auto seed_params_pair = sr_nloptimizer.global(params_buf);
            if (seed_params_pair.second && (seed_params.empty() || seed_params_pair.first(8) < seed_params(8))) {
                // This is either the first seed, or a seed which has a lower
                // correlation coefficient - so store it.
                seed_params = seed_params_pair.first;
            }
        }       

        return seed_params;
    }
        
    bool analyze_point(const Array2D<double> &queue_params,
                       ROI2D::difference_type p1_delta, 
                       ROI2D::difference_type p2_delta, 
                       const ROI2D::region_nlinfo &nlinfo,         
                       ROI2D::difference_type scalefactor,   
                       const subregion_nloptimizer &sr_nloptimizer,
                       double cutoff_corrcoef,           
                       double cutoff_delta_disp, 
                       std::priority_queue<Array2D<double>, std::vector<Array2D<double>>, std::function<bool(const Array2D<double>&,const Array2D<double>&)>> &queue, 
                       Array2D<bool> &A_ap, 
                       Array2D<double> &params_buf) {
        typedef ROI2D::difference_type                          difference_type; 
        
        // Note: params = {p1, p2, v, u, dv_dp1, dv_dp2, du_dp1, du_dp2, corr_coef, diff_norm}
                
        // Make sure points are in bounds of roi_reduced's nlinfo and havent 
        // been analyzed yet.
        difference_type p1 = queue_params(0) + p1_delta;
        difference_type p2 = queue_params(1) + p2_delta;
        if (nlinfo.in_nlinfo(p1 / scalefactor, p2 / scalefactor) &&
            A_ap(p1  / scalefactor, p2 / scalefactor)) {
            A_ap(p1 / scalefactor, p2 / scalefactor) = false; // Inactivate
                                    
            // Fill in params - get initial guess for displacements using 
            // gradients; just reuse gradients as their initial guess.
            params_buf(0) = p1;
            params_buf(1) = p2;
            params_buf(2) = queue_params(2) + p1_delta * queue_params(4) + p2_delta * queue_params(5);
            params_buf(3) = queue_params(3) + p1_delta * queue_params(6) + p2_delta * queue_params(7);
            params_buf(4) = queue_params(4);
            params_buf(5) = queue_params(5);
            params_buf(6) = queue_params(6);
            params_buf(7) = queue_params(7);

            // Get optimized params from nonlinear optimizer
            auto params_pair = sr_nloptimizer(params_buf);            
            if (params_pair.second) {
                // Insert into queue if it meets the criterion
                if (std::sqrt(std::pow(params_pair.first(2) - params_buf(2),2) + std::pow(params_pair.first(3) - params_buf(3),2)) < cutoff_delta_disp && 
                    params_pair.first(8) < cutoff_corrcoef) {
                    queue.push(params_pair.first);
                    
                    return true;
                }
            }             
            // Do not attempt global optimization if the guess fails since it 
            // is slow - just skip this point.
        }
        // Something failed
        return false;
    }
        
    void worker_RGDIC(subregion_nloptimizer sr_nloptimizer,     // passed by value
                      ROI2D roi_reduced,                        // passed by value
                      ROI2D::difference_type scalefactor,     
                      double cutoff_corrcoef,           
                      double cutoff_delta_disp, 
                      Array2D<double> &A_v,
                      Array2D<double> &A_u,
                      Array2D<double> &A_cc,
                      Array2D<bool> &A_ap,
                      Array2D<bool> &A_vp) {
        typedef ROI2D::difference_type                          difference_type;
                
        // Note: params = {p1, p2, v, u, dv_dp1, dv_dp2, du_dp1, du_dp2, corr_coef, diff_norm}
                       
        // For robustness, multiple seeds are placed within this region and the 
        // seed with the highest correlation coefficient is used in the 
        // analysis. The higher the number of seeds, the more robust (but slower)
        // the analysis will be.
        difference_type num_redundant_seeds = 12; 
        auto redundant_seeds = get_ROI_partition_diagram_seeds(get_ROI_partition_diagram(roi_reduced, num_redundant_seeds), roi_reduced, num_redundant_seeds);
        for (auto &region_seeds: redundant_seeds) {            
            // Must scale seed position
            region_seeds = std::move(region_seeds) * scalefactor; 
        }
                        
        // Cycle over regions and perform RGDIC.
        Array2D<double> params_buf(10, 1); // buffer for nloptimizer
        for (difference_type region_idx = 0; region_idx < roi_reduced.size_regions(); ++region_idx) {              
            if (redundant_seeds[region_idx].empty()) {
                // This region couldn't be seeded - most likely because it was 
                // too small.
                continue;
            }
            
            // Get seed params for queue
            auto seed_params = get_seed_params(redundant_seeds[region_idx],
                                               sr_nloptimizer, 
                                               params_buf);
            if (!seed_params.empty()) {    
                A_ap(seed_params(0) / scalefactor, seed_params(1) / scalefactor) = false;
                
                // Form queue - make it a priority queue such that the lowest 
                // correlation coefficient values are processed first.
                auto comp = [](const Array2D<double> &a, const Array2D<double> &b ) { return a(8) > b(8); };
                std::priority_queue<Array2D<double>, std::vector<Array2D<double>>, std::function<bool(const Array2D<double>&,const Array2D<double>&)>> queue(comp);

                // Perform flood fill around seed  
                queue.push(seed_params);                            
                while (!queue.empty()) {	                    
                    // 1) pop info from queue                
                    auto queue_params = std::move(queue.top()); queue.pop();

                    // 2) Validate and store information
                    A_vp(queue_params(0) / scalefactor,  queue_params(1) / scalefactor) = true;
                    A_v(queue_params(0) / scalefactor,  queue_params(1) / scalefactor) = queue_params(2);
                    A_u(queue_params(0) / scalefactor,  queue_params(1) / scalefactor) = queue_params(3);
                    A_cc(queue_params(0) / scalefactor,  queue_params(1) / scalefactor) = queue_params(8);
                                            
                    // 3) Analyze four surrounding points - up, down, left right;
                    analyze_point(queue_params, -scalefactor, 0, roi_reduced.get_nlinfo(region_idx), scalefactor, sr_nloptimizer, cutoff_corrcoef, cutoff_delta_disp, queue, A_ap, params_buf);
                    analyze_point(queue_params,  scalefactor, 0, roi_reduced.get_nlinfo(region_idx), scalefactor, sr_nloptimizer, cutoff_corrcoef, cutoff_delta_disp, queue, A_ap, params_buf);
                    analyze_point(queue_params, 0, -scalefactor, roi_reduced.get_nlinfo(region_idx), scalefactor, sr_nloptimizer, cutoff_corrcoef, cutoff_delta_disp, queue, A_ap, params_buf);
                    analyze_point(queue_params, 0,  scalefactor, roi_reduced.get_nlinfo(region_idx), scalefactor, sr_nloptimizer, cutoff_corrcoef, cutoff_delta_disp, queue, A_ap, params_buf);
                }    
            }
        }
                        
        // Clear all active points for this ROI as a way of indicating this 
        // thread has completed its work.
        A_ap(roi_reduced.get_mask()) = false;
    }    
}

std::pair<Disp2D, Data2D> RGDIC(const Array2D<double> &A_ref, 
                                const Array2D<double> &A_cur, 
                                const ROI2D &roi, 
                                ROI2D::difference_type scalefactor, 
                                INTERP interp_type, 
                                SUBREGION subregion_type, 
                                ROI2D::difference_type r, 
                                ROI2D::difference_type num_threads,
                                double cutoff_corrcoef,                
                                bool debug) { 
    typedef ROI2D::difference_type                              difference_type;
            
    if (!A_ref.same_size(A_cur)) {
        throw std::invalid_argument("Attempted to perform RGDIC on reference image input of size: " + A_ref.size_2D_string() + 
                                    " with current image input of size: " + A_cur.size_2D_string() + ". Sizes must be the same.");
    }
    
    if (!A_ref.same_size(roi.get_mask())) {
        throw std::invalid_argument("Attempted to perform RGDIC on reference image input of size: " + A_ref.size_2D_string() + 
                                    " with ROI input of size: " + roi.size_2D_string() + ". Sizes must be the same.");
    }
    
    if (scalefactor < 1) {
        throw std::invalid_argument("Attempted to perform RGDIC with scalefactor: " + std::to_string(scalefactor) +
                                    ". scalefactor must be 1 or greater.");
    }
    
    if (interp_type < INTERP::CUBIC_KEYS) {
        throw std::invalid_argument("Interpolation used for RGDIC must be cubic or greater.");
    }
    
    if (r < 5) {
        throw std::invalid_argument("Attempted to perform RGDIC with radius: " + std::to_string(r) +
                                    ". radius must be 5 or greater.");
    }
    
    if (num_threads < 1) {
        throw std::invalid_argument("Attempted to perform RGDIC with number of threads: " + std::to_string(num_threads) +
                                    ". Number of threads must be 1 or greater.");
    }
    
    if (cutoff_corrcoef < 0 || cutoff_corrcoef > 4.0) {
        throw std::invalid_argument("Input correlation coefficient cutoff of: " + std::to_string(cutoff_corrcoef) + 
                                    " is not between [0,4].");
    }
    
    // cutoff_delta_disp is a cutoff for the allowable change in displacement 
    // between analyzed points. Generally if a point is analyzed incorrectly,
    // it will result in a large change in displacement, so just filter these
    // points out.
    double cutoff_delta_disp = scalefactor;
    
    // Reduce ROI 
    auto roi_reduced = roi.reduce(scalefactor);
    
    // Get partition diagram of reduced ROI
    auto partition_diagram = details::get_ROI_partition_diagram(roi_reduced, num_threads);
            
    // Get subregion nonlinear optimizer
    auto sr_nloptimizer = details::subregion_nloptimizer(A_ref, A_cur, roi, scalefactor, interp_type, subregion_type, r);
    
    // Initialize displacement and correlation coefficient arrays
    Array2D<double> A_v(roi_reduced.height(), roi_reduced.width());
    Array2D<double> A_u(roi_reduced.height(), roi_reduced.width());
    Array2D<double> A_cc(roi_reduced.height(), roi_reduced.width());
    
    // -----------------------------------------------------------------------//
    // Perform reliability guided DIC ----------------------------------------//
    // -----------------------------------------------------------------------//    
    // Initialize buffers - note that only regions that were partitioned by
    // partition diagram can be analyzed, so set A_ap to all points which are 
    // in the partition diagram.
    Array2D<bool> A_ap(partition_diagram != -1);                   // Active Points
    Array2D<bool> A_vp(roi_reduced.height(), roi_reduced.width()); // Valid Points
    
    // Form threads 
    std::vector<std::thread> threads(num_threads);
    for (difference_type thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
        threads[thread_idx] = std::thread(details::worker_RGDIC,
                                          sr_nloptimizer,                                          // Send in copy of subregion nonlinear optimizer
                                          roi_reduced.form_union(partition_diagram == thread_idx), // Dont use std::ref since r-value
                                          scalefactor,
                                          cutoff_corrcoef,
                                          cutoff_delta_disp,
                                          std::ref(A_v),
                                          std::ref(A_u),
                                          std::ref(A_cc),
                                          std::ref(A_ap),
                                          std::ref(A_vp));
    }
    
    // Debugging stuff -------------------------------------------------------//    
    if (debug) {
        // These debugging tools are displayed during the RGDIC calculation and 
        // give real-time updates to help assist with analysis.
        
        // Display preview of a subset - display 1 per region
        auto subregion_gen = roi.get_contig_subregion_generator(subregion_type, r);
        auto subset_seeds = details::get_ROI_partition_diagram_seeds(details::get_ROI_partition_diagram(roi_reduced, 1), roi_reduced, 1);
        Array2D<double> subset(2*r+1,2*r+1);
        for (const auto &subset_seed : subset_seeds) {
            if (!subset_seed.empty()) {
                // Clear subset
                subset() = 0;                
                // Get subregion and the fill
                const auto &subregion_nlinfo = subregion_gen(subset_seed(0) * scalefactor, subset_seed(1) * scalefactor);                        
                for (difference_type nl_idx = 0; nl_idx < subregion_nlinfo.nodelist.width(); ++nl_idx) {
                    difference_type p2 = nl_idx + subregion_nlinfo.left_nl;
                    for (difference_type np_idx = 0; np_idx < subregion_nlinfo.noderange(nl_idx); np_idx += 2) {
                        difference_type np_top = subregion_nlinfo.nodelist(np_idx,nl_idx);
                        difference_type np_bottom = subregion_nlinfo.nodelist(np_idx + 1,nl_idx);
                        for (difference_type p1 = np_top; p1 <= np_bottom; ++p1) {   
                            difference_type p1_shifted = p1 - (subset_seed(0) * scalefactor) + subregion_gen.get_r();
                            difference_type p2_shifted = p2 - (subset_seed(1) * scalefactor) + subregion_gen.get_r();
                            // Set subset value
                            subset(p1_shifted, p2_shifted) = A_ref(p1,p2);
                        }
                    }
                }     
            }  
            // Display subset
            imshow(subset, 2000);
        }

        // Display partition diagram
        imshow(partition_diagram,2000);

        // Show v-plot getting updated - this acts as a waitbar. Typically 
        // deformation happens in the y-axis so this is a good plot to show
        // for updates. Note that threads clear out their portion of A_ap when
        // they are finished, so this condition will work correctly.
        while (any_true(A_ap)) {
            imshow(A_v,50);
        }
        imshow(A_v,50); 
    }
    // -----------------------------------------------------------------------//
    
    // Join threads
    for (difference_type thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
        threads[thread_idx].join();
    }    
    // -----------------------------------------------------------------------//    
    // -----------------------------------------------------------------------//    
    // -----------------------------------------------------------------------//    
    
    // Go back over plot and clear any datapoints that have large displacement
    // jumps. Not all are cleared through the RGDIC algorithm because of the 
    // way the search is conducted.
    auto A_vp_buf = A_vp; // Make copy of valid points since it gets overwritten
    for (difference_type region_idx = 0; region_idx < roi_reduced.size_regions(); ++region_idx) {
        for (difference_type nl_idx = 0; nl_idx < roi_reduced.get_nlinfo(region_idx).nodelist.width(); ++nl_idx) {
            difference_type p2 = nl_idx + roi_reduced.get_nlinfo(region_idx).left_nl;
            for (difference_type np_idx = 0; np_idx < roi_reduced.get_nlinfo(region_idx).noderange(nl_idx); np_idx += 2) {
                difference_type np_top = roi_reduced.get_nlinfo(region_idx).nodelist(np_idx,nl_idx);
                difference_type np_bottom = roi_reduced.get_nlinfo(region_idx).nodelist(np_idx + 1,nl_idx);
                for (difference_type p1 = np_top; p1 <= np_bottom; ++p1) {
                    if ((roi_reduced.get_nlinfo(region_idx).in_nlinfo(p1-1,p2) && A_vp_buf(p1-1,p2) && std::sqrt(std::pow(A_v(p1-1,p2) - A_v(p1,p2),2) + std::pow(A_u(p1-1,p2) - A_u(p1,p2),2)) > cutoff_delta_disp) ||
                        (roi_reduced.get_nlinfo(region_idx).in_nlinfo(p1+1,p2) && A_vp_buf(p1+1,p2) && std::sqrt(std::pow(A_v(p1+1,p2) - A_v(p1,p2),2) + std::pow(A_u(p1+1,p2) - A_u(p1,p2),2)) > cutoff_delta_disp) ||
                        (roi_reduced.get_nlinfo(region_idx).in_nlinfo(p1,p2-1) && A_vp_buf(p1,p2-1) && std::sqrt(std::pow(A_v(p1,p2-1) - A_v(p1,p2),2) + std::pow(A_u(p1,p2-1) - A_u(p1,p2),2)) > cutoff_delta_disp) ||
                        (roi_reduced.get_nlinfo(region_idx).in_nlinfo(p1,p2+1) && A_vp_buf(p1,p2+1) && std::sqrt(std::pow(A_v(p1,p2+1) - A_v(p1,p2),2) + std::pow(A_u(p1,p2+1) - A_u(p1,p2),2)) > cutoff_delta_disp)) {
                        A_vp(p1,p2) = false;
                    }
                }
            }
        }
    }            
    
    // Must form union with valid points
    auto roi_valid = roi_reduced.form_union(A_vp);
    
    return { Disp2D(std::move(A_v), std::move(A_u), roi_valid, scalefactor), Data2D(std::move(A_cc), roi_valid, scalefactor) };
}

DIC_analysis_input::DIC_analysis_input(const std::vector<Image2D> &imgs, 
                                       const ROI2D &roi,
                                       ROI2D::difference_type scalefactor, 
                                       INTERP interp_type,
                                       SUBREGION subregion_type,
                                       ROI2D::difference_type r,
                                       ROI2D::difference_type num_threads,
                                       DIC_analysis_config config_type,
                                       bool debug) : imgs(imgs),
                                                     roi(roi),
                                                     scalefactor(scalefactor),
                                                     interp_type(interp_type),
                                                     subregion_type(subregion_type), 
                                                     r(r), 
                                                     num_threads(num_threads), 
                                                     debug(debug) {         
    // Set parameters to some preset configuration
    switch (config_type) {
        case DIC_analysis_config::NO_UPDATE :
            // This will never update the reference image. Use this configuration
            // for images with low deformation.
            this->cutoff_corrcoef = 2.0;
            this->update_corrcoef = 4.0;  // Don't ever update
            this->prctile_corrcoef = 1.0;
            break;
        case DIC_analysis_config::KEEP_MOST_POINTS :
            // This will update as frequently as needed to keep as many points 
            // as possible. Use this configuration for samples undergoing large, 
            // continuous deformation.
            this->cutoff_corrcoef = 2.0;
            this->update_corrcoef = 0.5;
            this->prctile_corrcoef = 1.0; // Same as max()
            break;
        case DIC_analysis_config::REMOVE_BAD_POINTS :
            // This will update less frequently and attempt to remove poorly 
            // analyzed points. Use this configuration for samples which have 
            // large deformation with some discontinuous deformation. This will 
            // attempt to remove the points near the discontinuity.
            this->cutoff_corrcoef = 0.7;
            this->update_corrcoef = 0.35;
            this->prctile_corrcoef = 0.9;
            break;
    }
}  

DIC_analysis_input DIC_analysis_input::load(std::ifstream &is) {
    // Form empty DIC_analysis_input then fill in values in accordance to how they are saved
    DIC_analysis_input DIC_input;
    
    // Load Images
    difference_type num_images = 0;
    is.read(reinterpret_cast<char*>(&num_images), std::streamsize(sizeof(difference_type)));
    DIC_input.imgs.resize(num_images);
    for (auto &img : DIC_input.imgs) {
        img = Image2D::load(is);
    }
    
    // Load roi
    DIC_input.roi = ROI2D::load(is);
    
    // Load scalefactor
    is.read(reinterpret_cast<char*>(&DIC_input.scalefactor), std::streamsize(sizeof(difference_type)));
    
    // Load interp_type
    is.read(reinterpret_cast<char*>(&DIC_input.interp_type), std::streamsize(sizeof(INTERP)));
    
    // Load subregion_type
    is.read(reinterpret_cast<char*>(&DIC_input.subregion_type), std::streamsize(sizeof(SUBREGION)));
    
    // Load radius
    is.read(reinterpret_cast<char*>(&DIC_input.r), std::streamsize(sizeof(difference_type)));
    
    // Load number of threads
    is.read(reinterpret_cast<char*>(&DIC_input.num_threads), std::streamsize(sizeof(difference_type)));
    
    // Load cutoff_corrcoef
    is.read(reinterpret_cast<char*>(&DIC_input.cutoff_corrcoef), std::streamsize(sizeof(double)));
    
    // Load update_corrcoef
    is.read(reinterpret_cast<char*>(&DIC_input.update_corrcoef), std::streamsize(sizeof(double)));
    
    // Load prctile_corrcoef
    is.read(reinterpret_cast<char*>(&DIC_input.prctile_corrcoef), std::streamsize(sizeof(double)));
    
    // Load debug
    is.read(reinterpret_cast<char*>(&DIC_input.debug), std::streamsize(sizeof(bool)));
    
    return DIC_input;
}

DIC_analysis_input DIC_analysis_input::load(const std::string &filename) {
    // Form stream
    std::ifstream is(filename.c_str(), std::ios::in | std::ios::binary);
    if (!is.is_open()) {
        throw std::invalid_argument("Could not open " + filename + " for loading DIC_analysis_input.");
    }
    
    // Form DIC_analysis_input using stream static factory method
    auto DIC_input = DIC_analysis_input::load(is);
    
    // Close stream
    is.close();
    
    return DIC_input;
}
    
void save(const DIC_analysis_input &DIC_input, std::ofstream &os) {    
    typedef ROI2D::difference_type                              difference_type;
    
    // Save imgs -> roi -> scalefactor -> interp_type -> subregion_type -> 
    // radius -> num_threads -> cutoff_corrcoef -> update_corrcoef -> 
    // prctile_corrcoef -> debug
        
    difference_type num_imgs = DIC_input.imgs.size();
    os.write(reinterpret_cast<const char*>(&num_imgs), std::streamsize(sizeof(difference_type)));    
    for (const auto &img : DIC_input.imgs) {
        save(img, os);
    }
    
    save(DIC_input.roi, os);
    
    os.write(reinterpret_cast<const char*>(&DIC_input.scalefactor), std::streamsize(sizeof(difference_type)));   
    
    os.write(reinterpret_cast<const char*>(&DIC_input.interp_type), std::streamsize(sizeof(INTERP)));   
    
    os.write(reinterpret_cast<const char*>(&DIC_input.subregion_type), std::streamsize(sizeof(SUBREGION)));   
    
    os.write(reinterpret_cast<const char*>(&DIC_input.r), std::streamsize(sizeof(difference_type)));   
    
    os.write(reinterpret_cast<const char*>(&DIC_input.num_threads), std::streamsize(sizeof(difference_type)));   
    
    os.write(reinterpret_cast<const char*>(&DIC_input.cutoff_corrcoef), std::streamsize(sizeof(double)));      
    
    os.write(reinterpret_cast<const char*>(&DIC_input.update_corrcoef), std::streamsize(sizeof(double)));   
    
    os.write(reinterpret_cast<const char*>(&DIC_input.prctile_corrcoef), std::streamsize(sizeof(double)));   
    
    os.write(reinterpret_cast<const char*>(&DIC_input.debug), std::streamsize(sizeof(bool)));  
}

void save(const DIC_analysis_input &DIC_input, const std::string &filename) {
    // Form stream
    std::ofstream os(filename.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
    if (!os.is_open()) {
        throw std::invalid_argument("Could not open " + filename + " for saving DIC_analysis_input.");
    }
    
    // Save DIC_input into stream
    save(DIC_input, os);
        
    // Close stream
    os.close();
}

DIC_analysis_output DIC_analysis_output::load(std::ifstream &is) {
    // Form empty DIC_analysis_output then fill in values in accordance to how they are saved
    DIC_analysis_output DIC_output;
    
    // Load disps
    difference_type num_disps = 0;
    is.read(reinterpret_cast<char*>(&num_disps), std::streamsize(sizeof(difference_type)));
    DIC_output.disps.resize(num_disps);
    for (auto &disp : DIC_output.disps) {
        disp = Disp2D::load(is);
    }
    
    // Load perspective type
    is.read(reinterpret_cast<char*>(&DIC_output.perspective_type), std::streamsize(sizeof(PERSPECTIVE)));
    
    // Load units
    difference_type length = 0;
    is.read(reinterpret_cast<char*>(&length), std::streamsize(sizeof(difference_type)));
    DIC_output.units = std::string(length,' ');
    is.read(const_cast<char*>(DIC_output.units.c_str()), std::streamsize(length));
        
    // Load units per pixel
    is.read(reinterpret_cast<char*>(&DIC_output.units_per_pixel), std::streamsize(sizeof(double)));
    
    return DIC_output;
}

DIC_analysis_output DIC_analysis_output::load(const std::string &filename) {
    // Form stream
    std::ifstream is(filename.c_str(), std::ios::in | std::ios::binary);
    if (!is.is_open()) {
        throw std::invalid_argument("Could not open " + filename + " for loading DIC_analysis_output.");
    }
    
    // Form DIC_analysis_output using stream static factory method
    auto DIC_output = DIC_analysis_output::load(is);
    
    // Close stream
    is.close();
    
    return DIC_output;
}
    
void save(const DIC_analysis_output &DIC_output, std::ofstream &os) {    
    typedef ROI2D::difference_type                              difference_type;
    
    // Save disps
    difference_type num_disps = DIC_output.disps.size();
    os.write(reinterpret_cast<const char*>(&num_disps), std::streamsize(sizeof(difference_type)));    
    for (const auto &disp : DIC_output.disps) {
        save(disp, os);
    }    
    
    // Save perspective type
    os.write(reinterpret_cast<const char*>(&DIC_output.perspective_type), std::streamsize(sizeof(PERSPECTIVE)));   
        
    // Save units
    difference_type length = DIC_output.units.size();
    os.write(reinterpret_cast<const char*>(&length), std::streamsize(sizeof(difference_type)));
    os.write(DIC_output.units.c_str(), std::streamsize(length));
    
    // Save units per pixel
    os.write(reinterpret_cast<const char*>(&DIC_output.units_per_pixel), std::streamsize(sizeof(double))); 
}

void save(const DIC_analysis_output &DIC_output, const std::string &filename) {
    // Form stream
    std::ofstream os(filename.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
    if (!os.is_open()) {
        throw std::invalid_argument("Could not open " + filename + " for saving DIC_analysis_output.");
    }
    
    // Save DIC_output into stream
    save(DIC_output, os);
        
    // Close stream
    os.close();
}

DIC_analysis_output DIC_analysis(const DIC_analysis_input &DIC_input) {
    typedef ROI2D::difference_type                              difference_type;
            
    if (DIC_input.imgs.size() < 2) {
        // Must have at least two images for DIC analysis
        throw std::invalid_argument("Only " + std::to_string(DIC_input.imgs.size()) + " images provided to DIC_analysis(). 2 or more are required.");
    }
            
    // Start timer for entire analysis
    std::chrono::time_point<std::chrono::system_clock> start_analysis = std::chrono::system_clock::now();
    
    // Initialize output - note that output will be WRT the first (reference)
    // image which is assumed to be the Lagrangian perspective.
    DIC_analysis_output DIC_output;
    DIC_output.disps.resize(DIC_input.imgs.size()-1);
    DIC_output.perspective_type = PERSPECTIVE::LAGRANGIAN;
    DIC_output.units = "pixels";
    DIC_output.units_per_pixel = 1.0;
            
    // Set ROI for the reference image - this gets updated if reference image 
    // gets updated. Then, cycle over images and perform DIC.
    ROI2D roi_ref = DIC_input.roi; 
    for (difference_type ref_idx = 0, cur_idx = 1; cur_idx < difference_type(DIC_input.imgs.size()); ++cur_idx) {        
        // -------------------------------------------------------------------//
        // Perform RGDIC -----------------------------------------------------//
        // -------------------------------------------------------------------//
        std::cout << std::endl << "Processing displacement field " << cur_idx << " of " << DIC_input.imgs.size() - 1 << "." << std::endl;
        std::cout << "Reference image: " << DIC_input.imgs[ref_idx] << "." << std::endl;
        std::cout << "Current image: " << DIC_input.imgs[cur_idx] << "." << std::endl;
        
        std::chrono::time_point<std::chrono::system_clock> start_rgdic = std::chrono::system_clock::now();
        
        auto disp_pair = RGDIC(DIC_input.imgs[ref_idx].get_gs(), 
                               DIC_input.imgs[cur_idx].get_gs(), 
                               roi_ref, 
                               DIC_input.scalefactor, 
                               DIC_input.interp_type, 
                               DIC_input.subregion_type,
                               DIC_input.r, 
                               DIC_input.num_threads,
                               DIC_input.cutoff_corrcoef,
                               DIC_input.debug);
        
        std::chrono::time_point<std::chrono::system_clock> end_rgdic = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds_rgdic = end_rgdic - start_rgdic;
        std::cout << "Time: " << elapsed_seconds_rgdic.count() << "." << std::endl;
        // -------------------------------------------------------------------//
        // -------------------------------------------------------------------//
        // -------------------------------------------------------------------//
        
        // Store displacements
        if (ref_idx > 0) {
            // Must "add" displacements before storing if reference image has 
            // been updated.
            DIC_output.disps[cur_idx-1] = add(std::vector<Disp2D>({ DIC_output.disps[ref_idx-1], disp_pair.first }), DIC_input.interp_type);
        } else {
            DIC_output.disps[cur_idx-1] = disp_pair.first;
        }
        
        // Test to see if selected correlation coefficient value (based on input
        // "prctile_corrcoef") for the displacement plot exceeds correlation 
        // coefficient cutoff value; if it does, then update the reference image.
        Array2D<double> cc_values = disp_pair.second.get_array()(disp_pair.second.get_roi().get_mask());
        if (!cc_values.empty()) {
            double selected_corrcoef = prctile(cc_values, DIC_input.prctile_corrcoef);
            std::cout << "Selected correlation coefficient value: " << selected_corrcoef << ". Correlation coefficient update value: " << DIC_input.update_corrcoef << "." << std::endl;
            if (selected_corrcoef > DIC_input.update_corrcoef) {
                // Update the reference image index as well as the reference roi
                ref_idx = cur_idx;
                roi_ref = update(DIC_input.roi, DIC_output.disps[cur_idx-1], DIC_input.interp_type);
            }
        }     
    }
    
    // End timer for entire analysis
    std::chrono::time_point<std::chrono::system_clock> end_analysis = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds_analysis = end_analysis - start_analysis;
    std::cout << std::endl << "Total DIC analysis time: " << elapsed_seconds_analysis.count() << "." << std::endl;
        
    return DIC_output;
}

// Conversion between Lagrangian and Eulerian displacements ------------------//
namespace details {
    Disp2D update(const Disp2D &disp, INTERP interp_type) {         
        // Update v and u
        auto v_updated = update(disp.get_v(), disp, interp_type);
        auto u_updated = update(disp.get_u(), disp, interp_type);

        // Note that moving arrays is safe because update() creates new Data2Ds
        return { std::move(v_updated.get_array()), std::move(u_updated.get_array()), v_updated.get_roi(), disp.get_scalefactor() };
    }
}

DIC_analysis_output change_perspective(const DIC_analysis_output &DIC_output, INTERP interp_type) {
    typedef ROI2D::difference_type                              difference_type;
    
    if (DIC_output.perspective_type == PERSPECTIVE::EULERIAN) {
        // For now, do not support this perspective change, because by 
        // default, DIC_analysis() returns the displacement fields in the 
        // Lagrangian perspective, so this change would be redundant and less 
        // accurate.
        throw std::invalid_argument("Changing from Eulerian perspective back to Lagrangian is currently not supported. Just use the original output data from DIC_analysis().");
    }
        
    if (DIC_output.units != "pixels") {
        // Perform change_perspective() before applying units. This makes things 
        // simpler.
        throw std::invalid_argument("Changing from Lagrangian to Eulerian perspective must be done before applying units.");
    }
    
    // Initialize output
    DIC_analysis_output DIC_output_updated;     
    DIC_output_updated.perspective_type = PERSPECTIVE::EULERIAN; // This will be Eulerian since only Lagrangian inputs are supported for now.
    DIC_output_updated.units = DIC_output.units;    
    DIC_output_updated.units_per_pixel = DIC_output.units_per_pixel;
    
    // Cycle over displacement fields and convert
    std::cout << std::endl << "Changing perspective..." << std::endl;
    for (difference_type disp_idx = 0; disp_idx < difference_type(DIC_output.disps.size()); ++disp_idx) {
        std::cout << "Displacement field " << disp_idx+1 << " of " << DIC_output.disps.size() << "." << std::endl;       
        DIC_output_updated.disps.push_back(details::update(DIC_output.disps[disp_idx], interp_type));
    }
    
    return DIC_output_updated;
}

// set_units -----------------------------------------------------------------//
DIC_analysis_output set_units(const DIC_analysis_output &DIC_output, const std::string &units, double units_per_pixel) {
    typedef ROI2D::difference_type                              difference_type;
    
    if (DIC_output.units != "pixels") {
        // Units have already been set - do not do this twice for simplicity.
        throw std::invalid_argument("Units have already been set for this DIC_analysis_output.");        
    }
    
    // Initialize output
    DIC_analysis_output DIC_output_updated;   
    DIC_output_updated.perspective_type = DIC_output.perspective_type;
    DIC_output_updated.units = units;   
    DIC_output_updated.units_per_pixel = units_per_pixel;
    
    // Cycle over displacements and convert
    for (difference_type disp_idx = 0; disp_idx < difference_type(DIC_output.disps.size()); ++disp_idx) {
        // Get copies of displacement fields and apply conversion
        auto A_v = DIC_output.disps[disp_idx].get_v().get_array() * units_per_pixel;
        auto A_u = DIC_output.disps[disp_idx].get_u().get_array() * units_per_pixel;
        
        DIC_output_updated.disps.push_back(Disp2D(std::move(A_v), 
                                                  std::move(A_u), 
                                                  DIC_output.disps[disp_idx].get_roi(), 
                                                  DIC_output.disps[disp_idx].get_scalefactor()));
    }
    
    return DIC_output_updated;
}

// strain_analysis -----------------------------------------------------------//
Strain2D LS_strain(const Disp2D &disp, PERSPECTIVE perspective_type, double units_per_pixel, SUBREGION subregion_type, ROI2D::difference_type r) { 
    typedef ROI2D::difference_type                              difference_type;
        
    // This will calculate Green-Langrangian strains for displacement fields in
    // the Lagrangian perspective; Eulerian-Almansi strains for displacement
    // fields in the Eulerian perspective. Displacement gradients are calculated
    // using a least squares plane fit on a contiguous subregion of displacement
    // points, and then these values are used to calculate the strain.
    
    if (units_per_pixel <= 0) {
        throw std::invalid_argument("Attempted to calculate least squares strain with units_per_pixel: " + std::to_string(units_per_pixel) +
                                    ". units_per_pixel must be greater than zero.");
    }
    
    if (r < 1) {
        throw std::invalid_argument("Attempted to calculate least squares strain with radius: " + std::to_string(r) +
                                    ". radius must be 1 or greater.");
    }
    
    // Initialize outputs
    Array2D<double> A_eyy(disp.data_height(), disp.data_width());
    Array2D<double> A_exy(disp.data_height(), disp.data_width());
    Array2D<double> A_exx(disp.data_height(), disp.data_width());
    Array2D<bool> A_vp(disp.data_height(), disp.data_width());
    
    // Set buffers for the least squares plane fit
    Array2D<double> mat_LS(3,3);
    Array2D<double> v_LS(3,1);
    Array2D<double> u_LS(3,1);
    
    // Get contig subregion generator
    auto subregion_gen = disp.get_roi().get_contig_subregion_generator(subregion_type, r);
            
    // Cycle over entire roi and calculate strains
    for (difference_type region_idx = 0; region_idx < disp.get_roi().size_regions(); ++region_idx) {
        for (difference_type nl_idx = 0; nl_idx < disp.get_roi().get_nlinfo(region_idx).nodelist.width(); ++nl_idx) {
            difference_type p2 = nl_idx + disp.get_roi().get_nlinfo(region_idx).left_nl;
            for (difference_type np_idx = 0; np_idx < disp.get_roi().get_nlinfo(region_idx).noderange(nl_idx); np_idx += 2) {
                difference_type np_top = disp.get_roi().get_nlinfo(region_idx).nodelist(np_idx,nl_idx);
                difference_type np_bottom = disp.get_roi().get_nlinfo(region_idx).nodelist(np_idx + 1,nl_idx);
                for (difference_type p1 = np_top; p1 <= np_bottom; ++p1) {
                    // Reset buffers
                    mat_LS() = 0;
                    v_LS() = 0;
                    u_LS() = 0;
                    
                    // Get subregion nlinfo for input p1 and p2
                    const auto &subregion_nlinfo = subregion_gen(p1, p2);
                    
                    // Cycle over subregion and calculate mat_LS, u_LS, and v_LS
                    for (difference_type nl_idx_subregion = 0; nl_idx_subregion < subregion_nlinfo.nodelist.width(); ++nl_idx_subregion) {
                        difference_type p2_subregion = nl_idx_subregion + subregion_nlinfo.left_nl;
                        for (difference_type np_idx_subregion = 0; np_idx_subregion < subregion_nlinfo.noderange(nl_idx_subregion); np_idx_subregion += 2) {
                            difference_type np_top_subregion = subregion_nlinfo.nodelist(np_idx_subregion,nl_idx_subregion);
                            difference_type np_bottom_subregion = subregion_nlinfo.nodelist(np_idx_subregion + 1,nl_idx_subregion);
                            for (difference_type p1_subregion = np_top_subregion; p1_subregion <= np_bottom_subregion; ++p1_subregion) {     
                                difference_type delta_p1 = p1_subregion - p1;
                                difference_type delta_p2 = p2_subregion - p2;
                                double v = disp.get_v().get_array()(p1_subregion,p2_subregion);
                                double u = disp.get_u().get_array()(p1_subregion,p2_subregion);
                                                                
                                // LS matrix is symmetric so just fill lower half
                                mat_LS(0,0) += std::pow(delta_p1,2);
                                mat_LS(1,0) += delta_p1*delta_p2;
                                mat_LS(2,0) += delta_p1;
                                mat_LS(1,1) += std::pow(delta_p2,2);
                                mat_LS(2,1) += delta_p2;
                                mat_LS(2,2) += 1;
                                
                                v_LS(0) += delta_p1 * v;
                                v_LS(1) += delta_p2 * v;
                                v_LS(2) += v;

                                u_LS(0) += delta_p1 * u;
                                u_LS(1) += delta_p2 * u;
                                u_LS(2) += u;
                            }
                        }
                    }
                    
                    // Fill upperhalf of matrix
                    mat_LS(0,1) = mat_LS(1,0);
                    mat_LS(0,2) = mat_LS(2,0);
                    mat_LS(1,2) = mat_LS(2,1);
                              
                    // Solve for displacement gradients using Cholesky 
                    // decomposition since matrix should be symmetric positive
                    // definite.
                    auto mat_LS_linsolver = mat_LS.get_linsolver(LINSOLVER::CHOL);
                    if (mat_LS_linsolver) {
                        // Make copies of output since they will be scaled to
                        // account for scalefactor
                        auto dv_dp = mat_LS_linsolver.solve(v_LS);
                        auto du_dp = mat_LS_linsolver.solve(u_LS);
                        
                        // Must scale displacement gradients to account for the
                        // scalefactor and units per pixel
                        dv_dp = std::move(dv_dp) / disp.get_scalefactor() / units_per_pixel;
                        du_dp = std::move(du_dp) / disp.get_scalefactor() / units_per_pixel;
                                
                        // Now calculate strains based on configuration
                        switch (perspective_type) {
                            case PERSPECTIVE::LAGRANGIAN :
                                // This is Green-Lagrangian strain
                                A_eyy(p1,p2) = 0.5*(2*dv_dp(0) + std::pow(du_dp(0),2) + std::pow(dv_dp(0),2));
                                A_exy(p1,p2) = 0.5*(du_dp(0) + dv_dp(1) + du_dp(1)*du_dp(0) + dv_dp(1)*dv_dp(0));
                                A_exx(p1,p2) = 0.5*(2*du_dp(1) + std::pow(du_dp(1),2) + std::pow(dv_dp(1),2));
                                break;
                            case PERSPECTIVE::EULERIAN :
                                // This is Eulerian-Almansi strain
                                A_eyy(p1,p2) = 0.5*(2*dv_dp(0) - std::pow(du_dp(0),2) - std::pow(dv_dp(0),2));
                                A_exy(p1,p2) = 0.5*(du_dp(0) + dv_dp(1) - du_dp(1)*du_dp(0) - dv_dp(1)*dv_dp(0));
                                A_exx(p1,p2) = 0.5*(2*du_dp(1) - std::pow(du_dp(1),2) - std::pow(dv_dp(1),2));    
                                break;
                        }
                        // Point is valid
                        A_vp(p1,p2) = true;
                    }
                }
            }
        }
    }  
    // Must form union with valid points        
    return { std::move(A_eyy), std::move(A_exy), std::move(A_exx), disp.get_roi().form_union(A_vp), disp.get_scalefactor() };
}

strain_analysis_input strain_analysis_input::load(std::ifstream &is) {
    // Form empty strain_analysis_input then fill in values in accordance to how they are saved
    strain_analysis_input strain_input;
    
    // Load DIC_input
    strain_input.DIC_input = DIC_analysis_input::load(is);
    
    // Load DIC_output
    strain_input.DIC_output = DIC_analysis_output::load(is);
    
    // Load subregion type
    is.read(reinterpret_cast<char*>(&strain_input.subregion_type), std::streamsize(sizeof(SUBREGION)));
    
    // Load radius
    is.read(reinterpret_cast<char*>(&strain_input.r), std::streamsize(sizeof(difference_type)));
    
    return strain_input;
}

strain_analysis_input strain_analysis_input::load(const std::string &filename) {
    // Form stream
    std::ifstream is(filename.c_str(), std::ios::in | std::ios::binary);
    if (!is.is_open()) {
        throw std::invalid_argument("Could not open " + filename + " for loading strain_analysis_input.");
    }
    
    // Form strain_analysis_input using stream static factory method
    auto strain_input = strain_analysis_input::load(is);
    
    // Close stream
    is.close();
    
    return strain_input;
}
    
void save(const strain_analysis_input &strain_input, std::ofstream &os) {    
    typedef ROI2D::difference_type                              difference_type;
    
    // Save DIC_input -> DIC_output -> subregion_type -> r       
   
    save(strain_input.DIC_input, os);
    
    save(strain_input.DIC_output, os);    
    
    os.write(reinterpret_cast<const char*>(&strain_input.subregion_type), std::streamsize(sizeof(SUBREGION)));   
    
    os.write(reinterpret_cast<const char*>(&strain_input.r), std::streamsize(sizeof(difference_type)));   
}

void save(const strain_analysis_input &strain_input, const std::string &filename) {
    // Form stream
    std::ofstream os(filename.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
    if (!os.is_open()) {
        throw std::invalid_argument("Could not open " + filename + " for saving strain_analysis_input.");
    }
    
    // Save strain_input into stream
    save(strain_input, os);
        
    // Close stream
    os.close();
}

strain_analysis_output strain_analysis_output::load(std::ifstream &is) {
    // Form empty strain_analysis_output then fill in values in accordance to how they are saved
    strain_analysis_output strain_output;
    
    // Load strains
    difference_type num_strains = 0;
    is.read(reinterpret_cast<char*>(&num_strains), std::streamsize(sizeof(difference_type)));
    strain_output.strains.resize(num_strains);
    for (auto &strain : strain_output.strains) {
        strain = Strain2D::load(is);
    }
    
    return strain_output;
}

strain_analysis_output strain_analysis_output::load(const std::string &filename) {
    // Form stream
    std::ifstream is(filename.c_str(), std::ios::in | std::ios::binary);
    if (!is.is_open()) {
        throw std::invalid_argument("Could not open " + filename + " for loading strain_analysis_output.");
    }
    
    // Form strain_analysis_output using stream static factory method
    auto strain_output = strain_analysis_output::load(is);
    
    // Close stream
    is.close();
    
    return strain_output;
}
    
void save(const strain_analysis_output &strain_output, std::ofstream &os) {    
    typedef ROI2D::difference_type                              difference_type;
    
    // Save strains
    difference_type num_strains = strain_output.strains.size();
    os.write(reinterpret_cast<const char*>(&num_strains), std::streamsize(sizeof(difference_type)));    
    for (const auto &strain : strain_output.strains) {
        save(strain, os);
    }    
}

void save(const strain_analysis_output &strain_output, const std::string &filename) {
    // Form stream
    std::ofstream os(filename.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
    if (!os.is_open()) {
        throw std::invalid_argument("Could not open " + filename + " for saving strain_analysis_output.");
    }
    
    // Save strain_output into stream
    save(strain_output, os);
        
    // Close stream
    os.close();
}

strain_analysis_output strain_analysis(const strain_analysis_input &strain_input) {
    typedef ROI2D::difference_type                              difference_type;

    // Initialize output
    strain_analysis_output strain_output;

    // Cycle over each displacement field and calculate corresponding strains
    std::cout << std::endl << "Processing strain fields..." << std::endl;
    for (difference_type disp_idx = 0; disp_idx < difference_type(strain_input.DIC_output.disps.size()); ++disp_idx) {        
        std::cout << "Strain field " << disp_idx+1 << " of " << strain_input.DIC_output.disps.size() << "." << std::endl;        
        // Calculate strain field for this Disp2D
        strain_output.strains.push_back(LS_strain(strain_input.DIC_output.disps[disp_idx], 
                                                  strain_input.DIC_output.perspective_type,
                                                  strain_input.DIC_output.units_per_pixel,
                                                  strain_input.subregion_type, 
                                                  strain_input.r));
    }        
    
    return strain_output;
}

// Interface functions for viewing and saving data ---------------------------//
namespace details {
    cv::Mat cv_ncorr_data_over_img(const Image2D &img, 
                                   const Data2D &data, 
                                   double alpha, 
                                   double min_data, 
                                   double max_data, 
                                   bool enable_colorbar = true, 
                                   bool enable_axes = true,
                                   bool enable_scalebar = true,
                                   const std::string &units = "pixels",
                                   double units_per_pixel = 1.0,
                                   double num_units = -1.0,
                                   double font_size = 1.0,
                                   ROI2D::difference_type num_tick_marks = 11,
                                   int colormap = cv::COLORMAP_JET) {
        typedef ROI2D::difference_type                          difference_type;

        // Data2D's used with Ncorr interface functions use the p1 direction for
        // the y-axis (i.e.  pointing downwards) and the p2 direction for the 
        // x-axis (i.e pointing rightwards) with the origin at the top-left. 
        // This dictates how the axes are displayed and correspond to the data.
        // For general Data2D's x and y axis have no assumptions about their 
        // directions, so disable axes if general Data2D inputs are used.

        if (alpha < 0 || alpha > 1) {
            throw std::invalid_argument("alpha input for cv_ncorr_data_over_img() must be between 0 and 1.");
        }

        if (units_per_pixel <= 0) {
            throw std::invalid_argument("units_per_pixel input for cv_ncorr_data_over_img() must be greater than 0.");
        }

        if (num_units != -1 && num_units <= 0) {
            throw std::invalid_argument("num_units input for cv_ncorr_data_over_img() must be greater than 0.");
        }
        
        if (font_size <= 0) {
            throw std::invalid_argument("font_size input for cv_ncorr_data_over_img() must be greater than 0.");
        }

        if (num_tick_marks < 2) {
            throw std::invalid_argument("num_tick_marks input for cv_ncorr_data_over_img() must be greater than or equal to 2.");
        }    
        
        // Set up image first - must resize it to the data
        auto A_img = img.get_gs();    
        cv::Mat cv_img = get_cv_img(A_img, min(A_img), max(A_img));
        cv::resize(cv_img, cv_img, { int(data.data_width()), int(data.data_height()) });
        
        // Set up data - must apply color map.
        cv::Mat cv_data = get_cv_img(data.get_array(), min_data, max_data);
        cv::applyColorMap(cv_data, cv_data, colormap);

        // Set up plot by drawing the image with data over it
        difference_type border = 20;        
        difference_type plot_height = data.data_height() + 2*border;
        difference_type plot_width = data.data_width() + 2*border;
        cv::Mat cv_plot(plot_height, plot_width, CV_8UC3, cv::Vec3b(255,255,255)); // CV_8UC3 means 3-channel image        
        for (difference_type p2 = border; p2 < data.data_width() + border; ++p2) {
            difference_type p2_data = p2 - border;
            for (difference_type p1 = border; p1 < data.data_height() + border; ++p1) {
                difference_type p1_data = p1 - border;
                
                // Get data values
                auto data_rgb = cv_data.at<cv::Vec3b>(p1_data,p2_data);
                auto img_gs = cv_img.at<uchar>(p1_data,p2_data);

                if (data.get_roi()(p1_data,p2_data)) {
                    // Set data overlay if point is within ROI
                    cv_plot.at<cv::Vec3b>(p1,p2) = cv::Vec3b(img_gs*(1-alpha) + alpha*data_rgb.val[0], 
                                                             img_gs*(1-alpha) + alpha*data_rgb.val[1], 
                                                             img_gs*(1-alpha) + alpha*data_rgb.val[2]);
                } else {
                    // Just draw background image
                    cv_plot.at<cv::Vec3b>(p1,p2) = cv::Vec3b(img_gs, img_gs, img_gs);
                }
            }
        } 
        
        if (enable_colorbar) {   
            // Create color bar (if enabled) and horizontally concatenate it 
            // with cv_plot. 
                                    
            // These are opencv parameters for text drawing
            auto font_face = cv::FONT_HERSHEY_PLAIN;
            int font_thickness = 1;
            int baseline = 0;
            
            // Colorbar parameters - note that division by (num_tick_marks-1) is
            // safe because num_tick_marks has been checked to ensure it is 
            // greater than or equal to 2.
            difference_type num_chars = 8; // number of chars printed for colorbar tick labels
            difference_type text_offset_left = 5;
            difference_type colorbar_width = 20;       
            // Set colorbar background width based on widest string width
            difference_type colorbar_bg_width = 0;
            for (difference_type num_mark = 0; num_mark < num_tick_marks; ++num_mark) {
                auto text_width = cv::getTextSize(std::to_string(double(num_tick_marks-num_mark-1)/(num_tick_marks-1) * (min_data-max_data) + max_data).substr(0,num_chars), font_face, 0.75*font_size, font_thickness, &baseline).width;
                if (text_width > colorbar_bg_width) {
                    colorbar_bg_width = text_width;
                }
            } 
            colorbar_bg_width += colorbar_width + text_offset_left + border; // Finish width to account for spacing and offsets
            
            // Draw the color portion of colorbar first and then apply colormap
            cv::Mat cv_colorbar(plot_height, colorbar_bg_width, cv::DataType<uchar>::type);            
            for (difference_type p2 = 0; p2 < colorbar_width; ++p2) {
                for (difference_type p1 = border; p1 < data.data_height() + border; ++p1) {
                    // Applies value such that color will be max at the top of
                    // the colorbar after the colormap is applied.
                    cv_colorbar.at<uchar>(p1,p2) = double((data.data_height() + border - 1) - p1)/(data.data_height()-1) * 255;
                }
            } 
            cv::applyColorMap(cv_colorbar, cv_colorbar, colormap);
                        
            // Paint region outside colorbar white
            cv_colorbar(cv::Range(0,border),cv::Range::all()) = cv::Vec3b(255,255,255);
            cv_colorbar(cv::Range(data.data_height()+border,data.data_height()+2*border),cv::Range::all()) = cv::Vec3b(255,255,255);
            cv_colorbar(cv::Range::all(),cv::Range(colorbar_width,colorbar_bg_width)) = cv::Vec3b(255,255,255);
            
            // Paint black border inside colorbar
            cv_colorbar(cv::Range(border, border+1),cv::Range(0,colorbar_width)) = cv::Vec3b(0,0,0);
            cv_colorbar(cv::Range(data.data_height()+border-1,data.data_height()+border),cv::Range(0,colorbar_width)) = cv::Vec3b(0,0,0);
            cv_colorbar(cv::Range(border,data.data_height()+border),cv::Range(0,1)) = cv::Vec3b(0,0,0);
            cv_colorbar(cv::Range(border,data.data_height()+border),cv::Range(colorbar_width-1,colorbar_width)) = cv::Vec3b(0,0,0);
                        
            // Paint tick-marks and corresponding labels
            difference_type tick_mark_width = 4;
            for (difference_type num_mark = 0; num_mark < num_tick_marks; ++num_mark) {
                difference_type p1 = (num_mark*(data.data_height()-1))/(num_tick_marks-1) + border;
                
                // Paint left mark, then right mark
                cv_colorbar(cv::Range(p1, p1+1),cv::Range(0,tick_mark_width)) = cv::Vec3b(0,0,0);
                cv_colorbar(cv::Range(p1, p1+1),cv::Range(colorbar_width-tick_mark_width,colorbar_width)) = cv::Vec3b(0,0,0);
                
                // Draw tick mark value
                std::string tick_mark_label_str = std::to_string(num_mark*(min_data-max_data)/(num_tick_marks-1) + max_data).substr(0,num_chars);
                auto text_size = cv::getTextSize(tick_mark_label_str, font_face, 0.75*font_size, font_thickness, &baseline);
                cv::putText(cv_colorbar, 
                            tick_mark_label_str, 
                            cv::Point(colorbar_width + text_offset_left, num_mark*(data.data_height()-1)/(num_tick_marks-1)  + border + text_size.height/2),
                            font_face, 
                            0.75*font_size, 
                            cv::Scalar::all(0), 
                            font_thickness);
            }           
            
            // Horizontally concatenate with cv_plot
            cv::hconcat(cv_plot, cv_colorbar, cv_plot);
        }
        
        if (enable_axes) {   
            // axes_length is the longest dimension of the background axes
            difference_type axes_length = 0.25 * std::min(data.data_height(), data.data_width());
            
            // background axes is painted black
            cv::Point bg_axes_pts[1][8];
            bg_axes_pts[0][0] = cv::Point(border,                   border);
            bg_axes_pts[0][1] = cv::Point(border,                   border+axes_length);
            bg_axes_pts[0][2] = cv::Point(border+0.20*axes_length,  border+0.80*axes_length);
            bg_axes_pts[0][3] = cv::Point(border+0.10*axes_length,  border+0.80*axes_length);
            bg_axes_pts[0][4] = cv::Point(border+0.10*axes_length,  border+0.10*axes_length);
            bg_axes_pts[0][5] = cv::Point(border+0.80*axes_length,  border+0.10*axes_length);
            bg_axes_pts[0][6] = cv::Point(border+0.80*axes_length,  border+0.20*axes_length);
            bg_axes_pts[0][7] = cv::Point(border+axes_length,       border);
            const cv::Point* pts1[1] = { bg_axes_pts[0] };
            int npts1[] = { 8 };
            cv::fillPoly(cv_plot, pts1, npts1, 1, cv::Scalar(255,255,255));
            
            // foreground axes is painted white over the black background axes
            cv::Point fg_axes_pts[1][8];
            fg_axes_pts[0][0] = cv::Point(border,                   border);
            fg_axes_pts[0][1] = cv::Point(border,                   border+0.95*axes_length);
            fg_axes_pts[0][2] = cv::Point(border+0.125*axes_length, border+0.825*axes_length);
            fg_axes_pts[0][3] = cv::Point(border+0.070*axes_length, border+0.825*axes_length);
            fg_axes_pts[0][4] = cv::Point(border+0.070*axes_length, border+0.070*axes_length);
            fg_axes_pts[0][5] = cv::Point(border+0.825*axes_length, border+0.070*axes_length);
            fg_axes_pts[0][6] = cv::Point(border+0.825*axes_length, border+0.125*axes_length);
            fg_axes_pts[0][7] = cv::Point(border+0.95*axes_length,  border);
            const cv::Point* pts2[1] = { fg_axes_pts[0] };
            int npts2[] = { 8 };
            cv::fillPoly(cv_plot, pts2, npts2, 1, cv::Scalar(0,0,0));
            
            // Draw Y and X labels
            difference_type label_border = 5;
            difference_type label_offset = 5;
            double label_alpha = 0.5;
            
            // These are opencv parameters for text drawing            
            auto font_face = cv::FONT_HERSHEY_PLAIN;
            int font_thickness = 1;
            int baseline = 0;                  
            
            // Draw "Y" label
            std::string y_str = "Y";
            auto text_size_y = cv::getTextSize(y_str, font_face, font_size, font_thickness, &baseline);
            for (difference_type p2 = border+label_offset; p2 < border+label_offset+2.0*label_border+text_size_y.width; ++p2) {
                for (difference_type p1 = border+axes_length+label_offset; p1 < border+axes_length+label_offset+2.0*label_border+text_size_y.height; ++p1) {
                    if (p1 >= 0 && p1 < cv_plot.rows && 
                        p2 >= 0 && p2 < cv_plot.cols) {
                        auto cv_plot_rgb = cv_plot.at<cv::Vec3b>(p1,p2);
                
                        cv_plot.at<cv::Vec3b>(p1,p2) = cv::Vec3b(label_alpha*cv_plot_rgb.val[0], 
                                                                 label_alpha*cv_plot_rgb.val[1], 
                                                                 label_alpha*cv_plot_rgb.val[2]);
                    }
                }
            }
            cv::putText(cv_plot, 
                        y_str, 
                        cv::Point(border+label_offset+label_border, border+axes_length+label_offset+label_border+text_size_y.height),
                        font_face, 
                        font_size, 
                        cv::Scalar::all(255), 
                        font_thickness);
            
            // Draw "X" label
            std::string x_str = "X";
            auto text_size_x = cv::getTextSize(x_str, font_face, font_size, font_thickness, &baseline);
            for (difference_type p2 = border+axes_length+label_offset; p2 < border+axes_length+label_offset+2.0*label_border+text_size_x.width; ++p2) {
                for (difference_type p1 = border+label_offset; p1 < border+label_offset+2.0*label_border+text_size_x.height; ++p1) {
                    if (p1 >= 0 && p1 < cv_plot.rows && 
                        p2 >= 0 && p2 < cv_plot.cols) {
                        auto cv_plot_rgb = cv_plot.at<cv::Vec3b>(p1,p2);
                
                        cv_plot.at<cv::Vec3b>(p1,p2) = cv::Vec3b(label_alpha*cv_plot_rgb.val[0], 
                                                                 label_alpha*cv_plot_rgb.val[1], 
                                                                 label_alpha*cv_plot_rgb.val[2]);
                    }
                }
            }
            cv::putText(cv_plot, 
                        x_str, 
                        cv::Point(border+axes_length+label_offset+label_border, border+label_offset+label_border+text_size_x.height),
                        font_face, 
                        font_size, 
                        cv::Scalar::all(255), 
                        font_thickness);      
        }
                
        if (enable_scalebar) {   
            // Set scalebar dimensions
            difference_type scalebar_width = num_units == -1 ? data.data_width()/2 : num_units/units_per_pixel/data.get_scalefactor();
            difference_type scalebar_height = 5;
            
            // Set num_units if it isn't -1
            if (num_units == -1) {
                num_units = scalebar_width*units_per_pixel*data.get_scalefactor();
            }
            
            // Form text first so we can use its size in determining the scalebar
            // bg height.
            difference_type decimal_places = 2;
            std::stringstream ss_scalebar;
            ss_scalebar << std::fixed << std::setprecision(std::numeric_limits<double>::digits10) << num_units;
            std::string scalebar_str_unformatted = ss_scalebar.str();
            auto idx_period = scalebar_str_unformatted.find(".");
            std::string scalebar_str = idx_period == std::string::npos ? scalebar_str_unformatted : scalebar_str_unformatted.substr(0, idx_period + decimal_places + 1) + " " + units;
                        
            // These are opencv parameters for text drawing            
            auto font_face = cv::FONT_HERSHEY_PLAIN;
            int font_thickness = 1;
            int baseline = 0;            
            auto text_size = cv::getTextSize(scalebar_str, font_face, font_size, font_thickness, &baseline);   
            
            // Draw background first
            difference_type scalebar_bg_border = 10;
            difference_type scalebar_bg_width = scalebar_width + 2*scalebar_bg_border;
            difference_type scalebar_bg_height = scalebar_height + 3*scalebar_bg_border + text_size.height;
            difference_type scalebar_bg_offset = 10;
            double scalebar_bg_alpha = 0.5;
            for (difference_type p2 = border+scalebar_bg_offset; p2 < border+scalebar_bg_offset+scalebar_bg_width; ++p2) {
                for (difference_type p1 = border+data.data_height()-scalebar_bg_offset-scalebar_bg_height; p1 < border+data.data_height()-scalebar_bg_offset; ++p1) {                   
                    if (p1 >= 0 && p1 < cv_plot.rows && 
                        p2 >= 0 && p2 < cv_plot.cols) {
                        auto cv_plot_rgb = cv_plot.at<cv::Vec3b>(p1,p2);

                        cv_plot.at<cv::Vec3b>(p1,p2) = cv::Vec3b(scalebar_bg_alpha*cv_plot_rgb.val[0], 
                                                                 scalebar_bg_alpha*cv_plot_rgb.val[1], 
                                                                 scalebar_bg_alpha*cv_plot_rgb.val[2]);
                    }
                }
            }
                        
            // Draw scalebar
            for (difference_type p2 = border+scalebar_bg_offset+scalebar_bg_border; p2 < border+scalebar_bg_offset+scalebar_bg_border+scalebar_width; ++p2) {
                for (difference_type p1 = border+data.data_height()-scalebar_bg_offset-scalebar_bg_border-scalebar_height; p1 < border+data.data_height()-scalebar_bg_offset-scalebar_bg_border; ++p1) {
                    if (p1 >= 0 && p1 < cv_plot.rows && 
                        p2 >= 0 && p2 < cv_plot.cols) {
                        cv_plot.at<cv::Vec3b>(p1,p2) = cv::Vec3b(255,255,255);
                    }
                }
            }
            
            // Draw scalebar text          
            cv::putText(cv_plot, 
                        scalebar_str, 
                        cv::Point(border+scalebar_bg_offset+scalebar_bg_width/2.0-text_size.width/2.0, border+data.data_height()-scalebar_bg_offset-2*scalebar_bg_border-scalebar_height),
                        font_face, 
                        font_size, 
                        cv::Scalar::all(255), 
                        font_thickness);    
        }
        
        return cv_plot;
    }
}

void imshow_ncorr_data_over_img(const Image2D &img, const Data2D &data, ROI2D::difference_type delay) {          
    // Set parameters 
    double min_data = 0, max_data = 0; 
    Array2D<double> data_values = data.get_array()(data.get_roi().get_mask());
    if (!data_values.empty()) {
        min_data = prctile(data_values,0.01);
        max_data = prctile(data_values,0.99);
    }    
    
    // Get cv style matrix and show it - disable scalebar for simplicity since
    // that requires units and unit conversion parameters and this function is
    // mainly used for debugging purposes.
    auto cv_img = details::cv_ncorr_data_over_img(img, 
                                                  data, 
                                                  0.5,          // Alpha
                                                  min_data, 
                                                  max_data, 
                                                  true,         // enable_colorbar
                                                  true,         // enable_axes
                                                  false);       // enable_scalebar
    cv::imshow("Ncorr data", cv_img);
    delay == -1 ? cv::waitKey() : cv::waitKey(delay);
}

void save_ncorr_data_over_img(const std::string &filename, 
                              const Image2D &img,
                              const Data2D &data,
                              double alpha,
                              double min_data,
                              double max_data,
                              bool enable_colorbar,
                              bool enable_axes,
                              bool enable_scalebar,
                              const std::string &units,
                              double units_per_pixel,
                              double num_units,
                              double font_size,
                              ROI2D::difference_type num_tick_marks,
                              int colormap) {   
    // Get cv style matrix and save it
    auto cv_data_img = details::cv_ncorr_data_over_img(img, 
                                                       data, 
                                                       alpha,
                                                       min_data, 
                                                       max_data, 
                                                       enable_colorbar,
                                                       enable_axes,
                                                       enable_scalebar,
                                                       units,
                                                       units_per_pixel,
                                                       num_units,
                                                       font_size,
                                                       num_tick_marks,
                                                       colormap);
    cv::imwrite(filename, cv_data_img);
}

void save_ncorr_data_over_img_video(const std::string &filename, 
                                    const std::vector<Image2D> &imgs,
                                    const std::vector<Data2D> &data,
                                    double alpha,
                                    double fps,
                                    double min_data,
                                    double max_data,
                                    bool enable_colorbar,
                                    bool enable_axes,
                                    bool enable_scalebar,
                                    const std::string &units,
                                    double units_per_pixel,
                                    double num_units,
                                    double font_size,
                                    ROI2D::difference_type num_tick_marks,
                                    int colormap,
                                    double end_delay,
                                    int fourcc) {
    typedef ROI2D::difference_type                          difference_type;
    
    // Note that if only 1 image was input, interpret this as using the same 
    // image for each Data2D. If not, use the img the data corresponds to.
    
    if (imgs.size() != 1 && imgs.size() != data.size()) {        
        // Number of images must either be 1, or equal to the number of input data
        throw std::invalid_argument("Number of images used in save_data_over_img_video() must either be 1, or equal to the number of input data.");
    }
    
    if (data.empty()) {
        throw std::invalid_argument("Number of Data2D used in save_data_over_img_video() must be greater than or equal to 1.");
    }
    
    // All data must have the same size
    for (difference_type data_idx = 1; data_idx < difference_type(data.size()); ++data_idx) {
        if (data[data_idx].data_height() != data.front().data_height() ||
            data[data_idx].data_width() != data.front().data_width()) {
            throw std::invalid_argument("Attempted to use save_data_over_img_video() with data of differing sizes. All data must be the same size.");
        }
    }
    
    if (fps <= 0) {
        throw std::invalid_argument("fps input for save_data_over_img_video() must be greater than 0.");
    }
    
    if (end_delay < 0) {
        throw std::invalid_argument("end_delay input for save_data_over_img_video() must be greater than or equal to 0.");
    }    
    
    // All other parameters will be checked when calling cv_ncorr_data_over_img()

    std::cout << std::endl << "Saving video: " << filename << "..." << std::endl;

    // Initialize video    
    cv::VideoWriter output_video;  
    
    // Cycle over data and save
    for (difference_type data_idx = 0; data_idx < difference_type(data.size()); ++data_idx) {
        std::cout << "Frame " << data_idx+1 << " of " << data.size() << "." << std::endl;
        
        auto cv_data_img = details::cv_ncorr_data_over_img(imgs.size() == 1 ? imgs.front() : imgs[data_idx], 
                                                           data[data_idx], 
                                                           alpha,
                                                           min_data, 
                                                           max_data, 
                                                           enable_colorbar,
                                                           enable_axes,
                                                           enable_scalebar,
                                                           units,
                                                           units_per_pixel,
                                                           num_units,
                                                           font_size,
                                                           num_tick_marks,
                                                           colormap);
        
        if (data_idx == 0) {
            // Open video file - do this here because the size of the cv_data_img
            // depends on its inputs and the size is needed to open the video.
            output_video.open(filename, 
                              fourcc, 
                              fps, 
                              { cv_data_img.cols, cv_data_img.rows }, 
                              true);            

            if (!output_video.isOpened()) {
                throw std::invalid_argument("Cannot open video file: " + filename + " for save_data_over_img_video().");
            }
        }
        
        output_video << cv_data_img;
    }

    // Add the last frame again multiple times to provide a delay
    for (difference_type idx = 0; idx < fps*end_delay; ++idx) {
        output_video << details::cv_ncorr_data_over_img(imgs.size() == 1 ? imgs.front() : imgs.back(), 
                                                        data.back(), 
                                                        alpha,
                                                        min_data, 
                                                        max_data, 
                                                        enable_colorbar,
                                                        enable_axes,
                                                        enable_scalebar,
                                                        units,
                                                        units_per_pixel,
                                                        num_units,
                                                        font_size,
                                                        num_tick_marks,
                                                        colormap);
    }
}

void save_DIC_video(const std::string &filename, 
                    const DIC_analysis_input &DIC_input, 
                    const DIC_analysis_output &DIC_output, 
                    DISP disp_type,
                    double alpha,
                    double fps,
                    double min_disp,
                    double max_disp,
                    bool enable_colorbar,
                    bool enable_axes,
                    bool enable_scalebar,
                    double num_units,
                    double font_size,
                    ROI2D::difference_type num_tick_marks,
                    int colormap,
                    double end_delay,
                    int fourcc) {
    typedef ROI2D::difference_type                              difference_type;
                        
    // Use get_disp() function to obtain the specified displacement field
    std::function<const Data2D&(const Disp2D&)> get_disp;
    switch (disp_type) {
        case DISP::V : 
            get_disp = &Disp2D::get_v;
            break;
        case DISP::U : 
            get_disp = &Disp2D::get_u;
            break;            
    }   
                
    // Set min and max if they are NaN. They are set so that all data fits 
    // approximately between these bounds. This assumes maxima occur at the
    // first or last displacement plot for simplicity.
    Array2D<double> data_values_first = get_disp(DIC_output.disps.front()).get_array()(DIC_output.disps.front().get_roi().get_mask());
    Array2D<double> data_values_last = get_disp(DIC_output.disps.back()).get_array()(DIC_output.disps.back().get_roi().get_mask());
    if (std::isnan(min_disp) && !data_values_first.empty() && !data_values_last.empty()) {
        min_disp = std::min(prctile(data_values_first,0.01), prctile(data_values_last,0.01));
    }
    if (std::isnan(max_disp) && !data_values_first.empty() && !data_values_last.empty()) {
        max_disp = std::max(prctile(data_values_first,0.99), prctile(data_values_last,0.99));
    }
    
    // Get data plots
    std::vector<Data2D> data;
    for (difference_type disp_idx = 0; disp_idx < difference_type(DIC_output.disps.size()); ++disp_idx) {
        data.push_back(get_disp(DIC_output.disps[disp_idx]));
    }
    
    // Get imgs - this depends on perspective
    std::vector<Image2D> imgs;
    switch (DIC_output.perspective_type) {
        case PERSPECTIVE::LAGRANGIAN : 
            imgs.push_back(DIC_input.imgs.front());
            break;
        case PERSPECTIVE::EULERIAN : 
            for (difference_type img_idx = 1; img_idx < difference_type(DIC_input.imgs.size()); ++img_idx) { // Starts from 1
                imgs.push_back(DIC_input.imgs[img_idx]);
            }
            break;            
    }   
    
    // Save the video
    save_ncorr_data_over_img_video(filename, 
                                   imgs, 
                                   data, 
                                   alpha, 
                                   fps, 
                                   min_disp, 
                                   max_disp,  
                                   enable_colorbar, 
                                   enable_axes, 
                                   enable_scalebar, 
                                   DIC_output.units, 
                                   DIC_output.units_per_pixel,
                                   num_units,
                                   font_size,
                                   num_tick_marks,
                                   colormap,
                                   end_delay,
                                   fourcc);
}

void save_strain_video(const std::string &filename, 
                       const strain_analysis_input &strain_input, 
                       const strain_analysis_output &strain_output, 
                       STRAIN strain_type,
                       double alpha,
                       double fps, 
                       double min_strain,
                       double max_strain,
                       bool enable_colorbar,
                       bool enable_axes,
                       bool enable_scalebar,
                       double num_units,
                       double font_size,
                       ROI2D::difference_type num_tick_marks,
                       int colormap,
                       double end_delay,
                       int fourcc) {
    typedef ROI2D::difference_type                              difference_type;
                        
    // Use get_strain function to obtain the specified strain field
    std::function<const Data2D&(const Strain2D&)> get_strain;
    switch (strain_type) {
        case STRAIN::EYY : 
            get_strain = &Strain2D::get_eyy;
            break;
        case STRAIN::EXY : 
            get_strain = &Strain2D::get_exy;
            break;
        case STRAIN::EXX : 
            get_strain = &Strain2D::get_exx;
            break;       
    }   
        
    // Set min and max if they are NaN. They are set so that all data fits 
    // approximately between these bounds This assumes maxima occur at the
    // first or last strain plot for simplicity.
    Array2D<double> data_values_first = get_strain(strain_output.strains.front()).get_array()(strain_output.strains.front().get_roi().get_mask());
    Array2D<double> data_values_last = get_strain(strain_output.strains.back()).get_array()(strain_output.strains.back().get_roi().get_mask());
    if (std::isnan(min_strain) && !data_values_first.empty() && !data_values_last.empty()) {
        min_strain = std::min(prctile(data_values_first,0.01), prctile(data_values_last,0.01));
    }
    if (std::isnan(max_strain) && !data_values_first.empty() && !data_values_last.empty()) {
        max_strain = std::max(prctile(data_values_first,0.99), prctile(data_values_last,0.99));
    }
            
    // Get data plots
    std::vector<Data2D> data;
    for (difference_type strain_idx = 0; strain_idx < difference_type(strain_output.strains.size()); ++strain_idx) {
        data.push_back(get_strain(strain_output.strains[strain_idx]));
    }
    
    // Get imgs - this depends on perspective
    std::vector<Image2D> imgs;
    switch (strain_input.DIC_output.perspective_type) {
        case PERSPECTIVE::LAGRANGIAN : 
            imgs.push_back(strain_input.DIC_input.imgs.front());
            break;
        case PERSPECTIVE::EULERIAN : 
            for (difference_type img_idx = 1; img_idx < difference_type(strain_input.DIC_input.imgs.size()); ++img_idx) { // Starts from 1
                imgs.push_back(strain_input.DIC_input.imgs[img_idx]);
            }
            break;            
    }   
        
    // Save the video
    save_ncorr_data_over_img_video(filename, 
                                   imgs, 
                                   data, 
                                   alpha, 
                                   fps, 
                                   min_strain, 
                                   max_strain, 
                                   enable_colorbar, 
                                   enable_axes, 
                                   enable_scalebar, 
                                   strain_input.DIC_output.units, 
                                   strain_input.DIC_output.units_per_pixel,
                                   num_units,
                                   font_size,
                                   num_tick_marks,
                                   colormap,
                                   end_delay, 
                                   fourcc);
}

}
