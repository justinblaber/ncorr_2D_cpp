/* 
 * File:   ncorr.h
 * Author: justin
 *
 * Created on May 12, 2015, 1:33 AM
 */

#ifndef NCORR_H
#define	NCORR_H

#include "Array2D.h"
#include "Image2D.h"
#include "ROI2D.h"
#include "Data2D.h"
#include "Disp2D.h"
#include "Strain2D.h"

namespace ncorr {

namespace details {
    // Nonlinear optimization ------------------------------------------------//
    class nloptimizer_base {         
        public:      
            typedef std::ptrdiff_t                              difference_type;    
            typedef std::pair<difference_type,difference_type>           coords;  
                        
            // Rule of 5 and destructor --------------------------------------//
            nloptimizer_base() noexcept = default;
            nloptimizer_base(const nloptimizer_base&) = default;
            nloptimizer_base(nloptimizer_base&&) = default;
            nloptimizer_base& operator=(const nloptimizer_base&) = default;  
            nloptimizer_base& operator=(nloptimizer_base&&) = default;
            virtual ~nloptimizer_base() noexcept = default;
            
            // Additional Constructors ---------------------------------------//         
            nloptimizer_base(difference_type order, difference_type num_params) : 
                grad_buf(order,1), hess_buf(order,order), params(num_params,1) { }
                         
            // Arithmetic operations -----------------------------------------//
            std::pair<const Array2D<double>&, bool> global(const Array2D<double>&) const;
            std::pair<const Array2D<double>&, bool> operator()(const Array2D<double>&) const;           
            
        protected:       
            // Arithmetic operations -----------------------------------------//
            virtual bool initial_guess() const = 0;
            virtual bool iterative_search() const = 0;
            virtual bool newton() const = 0;
                         
            // Utility -------------------------------------------------------//
            void chk_input_params_size(const Array2D<double>&) const;
            
            mutable Array2D<double> grad_buf; 
            mutable Array2D<double> hess_buf; 
            mutable Array2D<double> params; 
            double cutoff_norm = 1e-6;
            difference_type cutoff_iterations = 50;
    };
    
    class disp_nloptimizer final : public nloptimizer_base {
        public:      
            typedef nloptimizer_base::difference_type           difference_type;   
            typedef nloptimizer_base::coords                             coords;
                        
            // Rule of 5 and destructor --------------------------------------//
            disp_nloptimizer() noexcept : region_idx() { }
            disp_nloptimizer(const disp_nloptimizer&) = default;
            disp_nloptimizer(disp_nloptimizer&&) = default;
            disp_nloptimizer& operator=(const disp_nloptimizer&) = default;  
            disp_nloptimizer& operator=(disp_nloptimizer&&) = default;
            ~disp_nloptimizer() noexcept = default;
            
            // Additional Constructors ---------------------------------------//         
            // Note: params = {p1_new, p2_new, p1_old, p2_old, v_old, u_old, dv_dp1_old, dv_dp2_old, du_dp1_old, du_dp2_old, dist, grad_norm}
            disp_nloptimizer(const Disp2D &disp, difference_type region_idx, INTERP interp_type) : 
                nloptimizer_base(2, 12), disp(disp), region_idx(region_idx), disp_interp(disp.get_nlinfo_interpolator(region_idx, interp_type)) { }
                         
        private:       
            // Arithmetic operations -----------------------------------------//
            bool initial_guess() const override;
            bool iterative_search() const override;
            bool newton() const override;
            
            Disp2D disp;                             // immutable - Disp2D has pointer semantics
            difference_type region_idx;
            Disp2D::nlinfo_interpolator disp_interp; // Have copy
    };
    
    class subregion_nloptimizer final : public nloptimizer_base {
        public:      
            typedef nloptimizer_base::difference_type           difference_type;   
            typedef nloptimizer_base::coords                             coords;
                        
            // Rule of 5 and destructor --------------------------------------//
            subregion_nloptimizer() noexcept : scalefactor(), ref_template_avg(), ref_template_ssd_inv() { }
            subregion_nloptimizer(const subregion_nloptimizer&) = default;
            subregion_nloptimizer(subregion_nloptimizer&&) = default;
            subregion_nloptimizer& operator=(const subregion_nloptimizer&) = default;  
            subregion_nloptimizer& operator=(subregion_nloptimizer&&) = default;
            ~subregion_nloptimizer() noexcept = default;
            
            // Additional Constructors ---------------------------------------//         
            // Note: params = {p1, p2, v, u, dv_dp1, dv_dp2, du_dp1, du_dp2, corr_coef, diff_norm}
            subregion_nloptimizer(const Array2D<double>&, const Array2D<double>&, const ROI2D&, difference_type, INTERP, SUBREGION, difference_type);
                         
        private:      
            // Arithmetic operations -----------------------------------------//
            bool initial_guess() const override;
            bool iterative_search() const override;
            bool newton() const override;
            
            std::shared_ptr<Array2D<double>> A_ref_ptr;                // Allows R-value arrays; immutable
            std::shared_ptr<Array2D<double>> A_cur_ptr;                // Allows R-value arrays; immutable
            difference_type scalefactor;
            Array2D<double>::interpolator A_cur_interp;                // Have copy
            ROI2D::contig_subregion_generator subregion_gen;           // Have copy
            // Buffers for NCC:
            std::shared_ptr<Array2D<double>> A_cur_cumsum_p1_ptr;      // immutable
            std::shared_ptr<Array2D<double>> A_cur_pow_cumsum_p1_ptr;  // immutable
            mutable Array2D<double> A_ref_template;                    // Have copy 
            // Buffers for inverse compositional gauss newton method:
            std::shared_ptr<Array2D<double>> A_dref_dp1_ptr;           // immutable
            std::shared_ptr<Array2D<double>> A_dref_dp2_ptr;           // immutable
            mutable double ref_template_avg;
            mutable double ref_template_ssd_inv;
            // Steepest descent images:
            mutable Array2D<double> A_dref_dv;                         // Have copy 
            mutable Array2D<double> A_dref_du;                         // Have copy 
            mutable Array2D<double> A_dref_dv_dp1;                     // Have copy 
            mutable Array2D<double> A_dref_dv_dp2;                     // Have copy 
            mutable Array2D<double> A_dref_du_dp1;                     // Have copy 
            mutable Array2D<double> A_dref_du_dp2;                     // Have copy 
            // Linsolver for hessian
            mutable Array2D<double>::linsolver hess_linsolver;         // Have copy 
            // Cur template buffer
            mutable Array2D<double> A_cur_template;                    // Have copy 
    };        
}

// Interface functions -------------------------------------------------------//
ROI2D update(const ROI2D&, const Disp2D&, INTERP);

Data2D update(const Data2D&, const Disp2D&, INTERP);

Disp2D add(const std::vector<Disp2D>&, INTERP);

// DIC_analysis --------------------------------------------------------------//
std::pair<Disp2D, Data2D> RGDIC(const Array2D<double>&, const Array2D<double>&, const ROI2D&, ROI2D::difference_type, INTERP, SUBREGION, ROI2D::difference_type, ROI2D::difference_type, double, bool);

enum class DIC_analysis_config { NO_UPDATE, KEEP_MOST_POINTS, REMOVE_BAD_POINTS };

struct DIC_analysis_input final {
    typedef ROI2D::difference_type                              difference_type;
        
    // Rule of 5 and destructor ----------------------------------------------//    
    DIC_analysis_input() : scalefactor(), interp_type(), subregion_type(), r(), num_threads(), cutoff_corrcoef(), update_corrcoef(), prctile_corrcoef(), debug() { }
    DIC_analysis_input(const DIC_analysis_input&) = default;
    DIC_analysis_input(DIC_analysis_input&&) = default;
    DIC_analysis_input& operator=(const DIC_analysis_input&) = default;
    DIC_analysis_input& operator=(DIC_analysis_input&&) = default;
    ~DIC_analysis_input() noexcept = default;
    
    // Additional constructors -----------------------------------------------//    
    DIC_analysis_input(const std::vector<Image2D> &imgs,
                       const ROI2D &roi,
                       difference_type scalefactor, 
                       INTERP interp_type,
                       SUBREGION subregion_type,
                       difference_type r,
                       difference_type num_threads,
                       double cutoff_corrcoef,
                       double update_corrcoef,
                       double prctile_corrcoef,
                       bool debug) : imgs(imgs),
                                     roi(roi),
                                     scalefactor(scalefactor),
                                     interp_type(interp_type),
                                     subregion_type(subregion_type), 
                                     r(r), 
                                     num_threads(num_threads), 
                                     cutoff_corrcoef(cutoff_corrcoef), 
                                     update_corrcoef(update_corrcoef),
                                     prctile_corrcoef(prctile_corrcoef), 
                                     debug(debug) { }    
    
    DIC_analysis_input(const std::vector<Image2D>&, const ROI2D&, difference_type, INTERP, SUBREGION, difference_type, difference_type, DIC_analysis_config, bool);
        
    // Static factory methods ------------------------------------------------//
    static DIC_analysis_input load(std::ifstream&);
    static DIC_analysis_input load(const std::string&);
            
    // Interface functions ---------------------------------------------------//
    friend void save(const DIC_analysis_input&, std::ofstream&);
    friend void save(const DIC_analysis_input&, const std::string&);
        
    std::vector<Image2D> imgs;
    ROI2D roi;
    difference_type scalefactor;
    INTERP interp_type;
    SUBREGION subregion_type; 
    difference_type r; 
    difference_type num_threads;
    double cutoff_corrcoef;
    double update_corrcoef;
    double prctile_corrcoef;
    bool debug;
};

enum class PERSPECTIVE { EULERIAN, LAGRANGIAN };

struct DIC_analysis_output final {
    typedef ROI2D::difference_type                              difference_type;
        
    // Rule of 5 and destructor ----------------------------------------------//    
    DIC_analysis_output() : units_per_pixel() { }
    DIC_analysis_output(const DIC_analysis_output&) = default;
    DIC_analysis_output(DIC_analysis_output&&) = default;
    DIC_analysis_output& operator=(const DIC_analysis_output&) = default;
    DIC_analysis_output& operator=(DIC_analysis_output&&) = default;
    ~DIC_analysis_output() noexcept = default;
    
    // Additional constructors -----------------------------------------------//    
    DIC_analysis_output(const std::vector<Disp2D> &disps, PERSPECTIVE perspective_type, const std::string &units, double units_per_pixel) : 
        disps(disps), perspective_type(perspective_type), units(units), units_per_pixel(units_per_pixel) { }
    
    // Static factory methods ------------------------------------------------//
    static DIC_analysis_output load(std::ifstream&);
    static DIC_analysis_output load(const std::string&);
            
    // Interface functions ---------------------------------------------------//
    friend void save(const DIC_analysis_output&, std::ofstream&);
    friend void save(const DIC_analysis_output&, const std::string&);
        
    std::vector<Disp2D> disps;
    PERSPECTIVE perspective_type;
    std::string units;
    double units_per_pixel;
};

DIC_analysis_output DIC_analysis(const DIC_analysis_input&);

// Conversion between Lagrangian and Eulerian displacements ------------------//
DIC_analysis_output change_perspective(const DIC_analysis_output&, INTERP);

// set units -----------------------------------------------------------------//
DIC_analysis_output set_units(const DIC_analysis_output&, const std::string&, double);

// strain_analysis -----------------------------------------------------------//
Strain2D LS_strain(const Disp2D&, PERSPECTIVE, double, SUBREGION, ROI2D::difference_type); 

struct strain_analysis_input final {
    typedef ROI2D::difference_type                              difference_type;
        
    // Rule of 5 and destructor ----------------------------------------------//    
    strain_analysis_input() : r() { }
    strain_analysis_input(const strain_analysis_input&) = default;
    strain_analysis_input(strain_analysis_input&&) = default;
    strain_analysis_input& operator=(const strain_analysis_input&) = default;
    strain_analysis_input& operator=(strain_analysis_input&&) = default;
    ~strain_analysis_input() noexcept = default;
    
    // Additional constructors -----------------------------------------------//    
    strain_analysis_input(const DIC_analysis_input &DIC_input,
                          const DIC_analysis_output &DIC_output,
                          SUBREGION subregion_type,
                          difference_type r) : DIC_input(DIC_input),
                                               DIC_output(DIC_output),
                                               subregion_type(subregion_type),
                                               r(r){ }   
    
    // Static factory methods ------------------------------------------------//
    static strain_analysis_input load(std::ifstream&);
    static strain_analysis_input load(const std::string&);
            
    // Interface functions ---------------------------------------------------//
    friend void save(const strain_analysis_input&, std::ofstream&);
    friend void save(const strain_analysis_input&, const std::string&);
        
    DIC_analysis_input DIC_input;
    DIC_analysis_output DIC_output;
    SUBREGION subregion_type; 
    difference_type r;
};

struct strain_analysis_output final {
    typedef ROI2D::difference_type                              difference_type;
        
    // Rule of 5 and destructor ----------------------------------------------//    
    strain_analysis_output() = default;
    strain_analysis_output(const strain_analysis_output&) = default;
    strain_analysis_output(strain_analysis_output&&) = default;
    strain_analysis_output& operator=(const strain_analysis_output&) = default;
    strain_analysis_output& operator=(strain_analysis_output&&) = default;
    ~strain_analysis_output() noexcept = default;
    
    // Additional constructors -----------------------------------------------//    
    strain_analysis_output(const std::vector<Strain2D> &strains) : strains(strains) { }
    
    // Static factory methods ------------------------------------------------//
    static strain_analysis_output load(std::ifstream&);
    static strain_analysis_output load(const std::string&);
            
    // Interface functions ---------------------------------------------------//
    friend void save(const strain_analysis_output&, std::ofstream&);
    friend void save(const strain_analysis_output&, const std::string&);
        
    std::vector<Strain2D> strains;
};

strain_analysis_output strain_analysis(const strain_analysis_input&);

// Interface functions for viewing and saving ncorr related data -------------//
void imshow_ncorr_data_over_img(const Image2D&, const Data2D&, ROI2D::difference_type = -1);

void save_ncorr_data_over_img(const std::string&, 
                              const Image2D&, 
                              const Data2D&, 
                              double, 
                              double, 
                              double, 
                              bool, 
                              bool, 
                              bool, 
                              const std::string&, 
                              double, 
                              double, 
                              double,
                              ROI2D::difference_type,
                              int);

void save_ncorr_data_over_img_video(const std::string&, 
                                    const std::vector<Image2D>&, 
                                    const std::vector<Data2D>&, 
                                    double, 
                                    double, 
                                    double, 
                                    double, 
                                    bool, 
                                    bool, 
                                    bool, 
                                    const std::string&, 
                                    double, 
                                    double, 
                                    double,
                                    ROI2D::difference_type,
                                    int, 
                                    double, 
                                    int);

enum class DISP { U, V };
void save_DIC_video(const std::string&, 
                    const DIC_analysis_input&, 
                    const DIC_analysis_output&, 
                    DISP, 
                    double, 
                    double, 
                    double = std::numeric_limits<double>::quiet_NaN(), 
                    double = std::numeric_limits<double>::quiet_NaN(), 
                    bool = true, 
                    bool = true, 
                    bool = true, 
                    double = -1.0,
                    double = 1.0, 
                    ROI2D::difference_type = 11,
                    int = cv::COLORMAP_JET,
                    double = 2.0, 
                    int = cv::VideoWriter::fourcc('M','J','P','G'));

enum class STRAIN { EYY, EXY, EXX };
void save_strain_video(const std::string&, 
                       const strain_analysis_input&, 
                       const strain_analysis_output&, 
                       STRAIN, 
                       double, 
                       double, 
                       double = std::numeric_limits<double>::quiet_NaN(), 
                       double = std::numeric_limits<double>::quiet_NaN(), 
                       bool = true, 
                       bool = true, 
                       bool = true, 
                       double = -1.0,
                       double = 1.0, 
                       ROI2D::difference_type = 11,
                       int = cv::COLORMAP_JET,
                       double = 2.0, 
                       int = cv::VideoWriter::fourcc('M','J','P','G'));
// ---------------------------------------------------------------------------//

}

#endif	/* NCORR_H */