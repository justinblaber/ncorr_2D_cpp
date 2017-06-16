/* 
 * File:   ROI2D.h
 * Author: justin
 *
 * Created on February 11, 2015, 12:02 AM
 */

#ifndef ROI2D_H
#define	ROI2D_H

#include "Array2D.h"

namespace ncorr {    
    
namespace details {
    // incrementor is like an iterator, except it does not have the dereference operation
    class nlinfo_incrementor;
    class ROI2D_incrementor;    
        
    class ROI2D_contig_subregion_generator; // Forms a contiguous subregion around an (x,y) input
}

enum class SUBREGION { CIRCLE, SQUARE };

class ROI2D final { 
public:      
    typedef std::ptrdiff_t                                       difference_type;   
    typedef std::pair<difference_type, difference_type>                   coords;  
    typedef details::ROI2D_incrementor                               incrementor; 
    typedef details::ROI2D_contig_subregion_generator contig_subregion_generator; 
    
    struct region_boundary;
    struct region_nlinfo;
    struct region;
    
    friend incrementor;
    friend contig_subregion_generator;
    
    // Rule of 5 and destructor ----------------------------------------------//        
    ROI2D() : points() { }
    ROI2D(const ROI2D&) = default;
    ROI2D(ROI2D&&) noexcept = default;
    ROI2D& operator=(const ROI2D&) = default;
    ROI2D& operator=(ROI2D&&) = default;  
    ~ROI2D() noexcept = default;
    
    // Additional Constructors -----------------------------------------------//
    explicit ROI2D(Array2D<bool>, difference_type = 0); // by-value
    ROI2D(region_nlinfo, difference_type, difference_type); // by-value
    ROI2D(region_boundary, difference_type, difference_type); // by-value
    ROI2D(std::vector<region_nlinfo>, difference_type, difference_type); // by-value
    ROI2D(std::vector<region_boundary>, difference_type, difference_type); // by-value
    
    // Static factory methods ------------------------------------------------//
    static ROI2D simple_circle(difference_type);
    static ROI2D simple_square(difference_type);
    static ROI2D load(std::ifstream&);
    
    // Operators interface ---------------------------------------------------//
    friend std::ostream& operator<<(std::ostream&, const ROI2D&);  
    friend void imshow(const ROI2D &roi, difference_type delay = -1) { imshow(*roi.mask_ptr, delay); }  
    friend bool isequal(const ROI2D&, const ROI2D&);
    friend void save(const ROI2D&, std::ofstream&);   
    
    // Access ----------------------------------------------------------------//
    // Note that ROI2D is immutable, so all access should be const.   
    difference_type height() const { return mask_ptr->height(); } 
    difference_type width() const { return mask_ptr->width(); } 
    bool empty() const { return points == 0; }
    bool in_bounds(difference_type p) const { return mask_ptr->in_bounds(p); }
    bool in_bounds(difference_type p1, difference_type p2) const { return mask_ptr->in_bounds(p1,p2); }
    
    bool operator()(difference_type p) const { return (*mask_ptr)(p); }
    bool operator()(difference_type p1, difference_type p2) const { return (*mask_ptr)(p1,p2); }
    // Perhaps add forwarding of Array2D style region indexing later.
    
    const Array2D<bool>& get_mask() const { return *mask_ptr; }; 
    const region_nlinfo& get_nlinfo(difference_type) const;
    const region_boundary& get_boundary(difference_type) const; 
    difference_type get_points() const { return points; }     
    difference_type size_regions() const { return regions_ptr->size(); }
    std::pair<difference_type,difference_type> get_region_idx(difference_type, difference_type) const;
                       
    // Arithmetic operations -------------------------------------------------//
    ROI2D reduce(difference_type) const;
    ROI2D form_union(const Array2D<bool>&) const;
        
    // incrementor -----------------------------------------------------------//
    incrementor begin_inc() const;
    incrementor end_inc() const;
    
    // contig_subregion_generator --------------------------------------------//
    contig_subregion_generator get_contig_subregion_generator(SUBREGION, difference_type) const;
    
    // Utility ---------------------------------------------------------------//
    std::string size_string() const { return mask_ptr->size_string(); }   
    std::string size_2D_string() const { return mask_ptr->size_2D_string(); }   
        
private:
    // Utility functions -----------------------------------------------------//   
    void set_points();
    void draw_mask();
    
    // Checks ----------------------------------------------------------------//
    void chk_region_idx_in_bounds(difference_type, const std::string&) const;
    
    std::shared_ptr<Array2D<bool>> mask_ptr;            // immutable
    std::shared_ptr<std::vector<region>> regions_ptr;   // immutable
    difference_type points;
};

struct ROI2D::region_nlinfo final { 
    typedef details::nlinfo_incrementor                             incrementor; 
    
    // nlinfo maintains 4-way contiguity as invariant
            
    // Rule of 5 and destructor ----------------------------------------------//    
    region_nlinfo() : top(), bottom(), left(), right(), left_nl(), right_nl(), points() { }
    region_nlinfo(const region_nlinfo&) = default;
    region_nlinfo(region_nlinfo&&) = default;
    region_nlinfo& operator=(const region_nlinfo&) = default;
    region_nlinfo& operator=(region_nlinfo&&) = default;
    ~region_nlinfo() noexcept = default;
    
    friend ROI2D;
    
    // Additional constructor ------------------------------------------------//    
    region_nlinfo(difference_type top, 
                  difference_type bottom,
                  difference_type left, 
                  difference_type right, 
                  difference_type nl_left,
                  difference_type nl_right,
                  difference_type h_nl,
                  difference_type w_nl,
                  difference_type points) : top(top),
                                            bottom(bottom),
                                            left(left),
                                            right(right),
                                            left_nl(nl_left), 
                                            right_nl(nl_right), 
                                            nodelist(h_nl,w_nl), 
                                            noderange(1,w_nl), 
                                            points(points) { }      
    
    // Static factory methods ------------------------------------------------//
    static region_nlinfo load(std::ifstream&);
        
    // Operators interface ---------------------------------------------------//
    friend std::ostream& operator<<(std::ostream&, const ROI2D::region_nlinfo&); 
    friend bool isequal(const region_nlinfo&, const region_nlinfo&); 
    friend void save(const region_nlinfo&, std::ofstream&);   
        
    // Access methods --------------------------------------------------------//    
    // The following 6 operations are inlined and defined in this header file.
    difference_type first_pos_idx() const;
    difference_type first_pos_p1() const;
    difference_type first_pos_p2() const;
    difference_type last_pos_idx() const;
    difference_type last_pos_p1() const;
    difference_type last_pos_p2() const;
    
    bool empty() const { return points == 0; }
    
    // Arithmetic methods ----------------------------------------------------//
    bool in_nlinfo(difference_type, difference_type) const;
    region_nlinfo& shift(difference_type, difference_type); // in-place
        
    // Incrementor -----------------------------------------------------------//
    incrementor begin_inc() const;
    incrementor end_inc() const;
    
    // Static factory methods ------------------------------------------------//
    static std::pair<std::vector<ROI2D::region_nlinfo>,bool> form_nlinfos(const Array2D<bool>&, ROI2D::difference_type = 0);
    
    difference_type top;
    difference_type bottom;
    difference_type left;
    difference_type right;
    difference_type left_nl;  // left p2 position of beginning of nodelist - can differ from "left"
    difference_type right_nl; // right p2 position of end of nodelist - can differ from "right"
    Array2D<difference_type> nodelist;
    Array2D<difference_type> noderange;
    difference_type points;
    
private:
    // Arithmetic methods ----------------------------------------------------//
    // Note that operations on nlinfo must maintain 4-way contiguity invariant
    region_nlinfo reduce(difference_type, Array2D<bool>&) const;
    region_nlinfo form_union(const Array2D<bool>&, Array2D<bool>&) const;
    region_nlinfo largest_contig_nlinfo(Array2D<bool>&) const;
    ROI2D::region_boundary to_boundary(Array2D<bool>&) const;
    
    // Utility ---------------------------------------------------------------//
    void chk_nonempty_op(const std::string&) const;
};

struct ROI2D::region_boundary final {    
    // Rule of 5 and destructor ----------------------------------------------//  
    region_boundary() = default;
    region_boundary(const region_boundary&) = default;
    region_boundary(region_boundary&&) = default;
    region_boundary& operator=(const region_boundary&) = default;
    region_boundary& operator=(region_boundary&&) = default;
    ~region_boundary() noexcept = default;
    
    friend ROI2D;
            
    // Additional Constructors -----------------------------------------------//
    region_boundary(const Array2D<double> &add, const std::vector<Array2D<double>> &sub) : add(add), sub(sub) { }
    region_boundary(const Array2D<double> &add, std::vector<Array2D<double>> &&sub) : add(add), sub(std::move(sub)) { }
    region_boundary(Array2D<double> &&add, const std::vector<Array2D<double>> &sub) : add(std::move(add)), sub(sub) { }
    region_boundary(Array2D<double> &&add, std::vector<Array2D<double>> &&sub) : add(std::move(add)), sub(std::move(sub)) { }
            
    // Static factory methods ------------------------------------------------//
    static region_boundary load(std::ifstream&);
    
    // Operators interface ---------------------------------------------------//
    friend std::ostream& operator<<(std::ostream&, const ROI2D::region_boundary&); 
    friend bool isequal(const region_boundary&, const region_boundary&);     
    friend void save(const region_boundary&, std::ofstream&);   
    
    Array2D<double> add;
    std::vector<Array2D<double>> sub;
    
private:
    // Arithmetic methods ----------------------------------------------------//
    ROI2D::region_nlinfo to_nlinfo(Array2D<bool>&) const;        
};
 
struct ROI2D::region final {           
    // Rule of 5 and destructor ----------------------------------------------//  
    region() = default;
    region(const region&) = default;
    region(region&&) = default;
    region& operator=(const region&) = default;
    region& operator=(region&&) = default;
    ~region() noexcept = default;
    
    // Additional Constructors -----------------------------------------------//
    region(const region_nlinfo &nlinfo, const region_boundary &boundary) : nlinfo(nlinfo), boundary(boundary) { }
    region(const region_nlinfo &nlinfo, region_boundary &&boundary) : nlinfo(nlinfo), boundary(std::move(boundary)) { }
    region(region_nlinfo &&nlinfo, const region_boundary &boundary) : nlinfo(std::move(nlinfo)), boundary(boundary) { }
    region(region_nlinfo &&nlinfo, region_boundary &&boundary) : nlinfo(std::move(nlinfo)), boundary(std::move(boundary)) { }
    
    // Static factory methods ------------------------------------------------//
    static region load(std::ifstream&);
    
    // Operators interface ---------------------------------------------------//
    friend std::ostream& operator<<(std::ostream&, const ROI2D::region&);   
    friend bool isequal(const region &reg1, const region &reg2) { return isequal(reg1.nlinfo,reg2.nlinfo) && isequal(reg1.boundary,reg2.boundary); }  
    friend void save(const region&, std::ofstream&);   
    
    region_nlinfo nlinfo;
    region_boundary boundary;
};

namespace details {     
    // Incrementors ----------------------------------------------------------//
    class nlinfo_incrementor final {    
        public:     
            typedef ROI2D::difference_type                      difference_type;  
            typedef ROI2D::coords                                        coords;  
            
            friend ROI2D::region_nlinfo; 
            
            // Rule of 5 and destructor --------------------------------------//        
            nlinfo_incrementor() noexcept : nlinfo_ptr(nullptr), nl_idx(), np_idx(), p1() { }
            nlinfo_incrementor(const nlinfo_incrementor&) = default;
            nlinfo_incrementor(nlinfo_incrementor&&) = default;
            nlinfo_incrementor& operator=(const nlinfo_incrementor&) = default;
            nlinfo_incrementor& operator=(nlinfo_incrementor&&) = default;
            ~nlinfo_incrementor() noexcept = default;
                    
            // Additional Constructors ---------------------------------------//
            nlinfo_incrementor(const ROI2D::region_nlinfo &nlinfo, difference_type nl_idx, difference_type np_idx, difference_type p1) : 
                nlinfo_ptr(&nlinfo), nl_idx(nl_idx), np_idx(np_idx), p1(p1) { }
            
            // Access methods ------------------------------------------------//
            coords pos_2D() const { return { p1, nlinfo_ptr->left_nl + nl_idx }; }

            // Arithmetic methods --------------------------------------------//
            nlinfo_incrementor& operator++();
            // Maybe add decrement operator later
            
            bool operator==(const nlinfo_incrementor &inc) const {
                return inc.nlinfo_ptr == nlinfo_ptr && inc.nl_idx == nl_idx && inc.np_idx == np_idx && inc.p1 == p1;
            }
            bool operator!=(const nlinfo_incrementor &inc) const { return !(inc == *this); }
            
        private:
            const ROI2D::region_nlinfo *nlinfo_ptr;
            difference_type nl_idx;  
            difference_type np_idx;  
            difference_type p1;          
    };  
    
    class ROI2D_incrementor final {       
        public:     
            typedef ROI2D::difference_type                      difference_type;  
            typedef ROI2D::coords                                        coords;  
            
            friend ROI2D;
            
            // Rule of 5 and destructor --------------------------------------//        
            ROI2D_incrementor() noexcept : region_idx() { }
            ROI2D_incrementor(const ROI2D_incrementor&) = default;
            ROI2D_incrementor(ROI2D_incrementor&&) = default;
            ROI2D_incrementor& operator=(const ROI2D_incrementor&) = default;
            ROI2D_incrementor& operator=(ROI2D_incrementor&&) = default;
            ~ROI2D_incrementor() noexcept = default;
                    
            // Additional Constructors ---------------------------------------//
            ROI2D_incrementor(const ROI2D&, difference_type, const ROI2D::region_nlinfo::incrementor&);
            
            // Access methods ------------------------------------------------//
            coords pos_2D() const { return nlinfo_inc.pos_2D(); }

            // Arithmetic methods --------------------------------------------//
            ROI2D_incrementor& operator++();
            // Maybe add decrement operator later
            
            bool operator==(const ROI2D_incrementor &inc) const {
                return inc.region_idx == region_idx && inc.nlinfo_inc == nlinfo_inc;
            }
            bool operator!=(const ROI2D_incrementor &inc) const { return !(*this == inc); }
                        
        private:          
            ROI2D roi; // ROI2D has pointer semantics
            difference_type region_idx;      
            ROI2D::region_nlinfo::incrementor nlinfo_inc;
    };
    
    // contig_subregion_generator --------------------------------------------//
    class ROI2D_contig_subregion_generator final {       
        public:     
            typedef ROI2D::difference_type                      difference_type;  
            
            friend ROI2D;
            
            // Rule of 5 and destructor --------------------------------------//        
            ROI2D_contig_subregion_generator() noexcept : r() { }
            ROI2D_contig_subregion_generator(const ROI2D_contig_subregion_generator&) = default;
            ROI2D_contig_subregion_generator(ROI2D_contig_subregion_generator&&) = default;
            ROI2D_contig_subregion_generator& operator=(const ROI2D_contig_subregion_generator&) = default;
            ROI2D_contig_subregion_generator& operator=(ROI2D_contig_subregion_generator&&) = default;
            ~ROI2D_contig_subregion_generator() noexcept = default;
                                
            // Additional Constructors ---------------------------------------//
            ROI2D_contig_subregion_generator(const ROI2D&, SUBREGION, difference_type);
            
            // Access methods ------------------------------------------------//
            difference_type get_r() const { return r; }
            const ROI2D::region_nlinfo& get_subregion_nlinfo() const { return nlinfo_subregion; }
            
            // Arithmetic methods --------------------------------------------//
            const ROI2D::region_nlinfo& operator()(difference_type, difference_type) const;
            
        private:    
            ROI2D roi; // ROI2D has pointer semantics
            difference_type r;
            mutable Array2D<bool> active_nodepairs;        
            mutable ROI2D::region_nlinfo nlinfo_simple;    
            mutable ROI2D::region_nlinfo nlinfo_subregion; 
    };
}

inline ROI2D::difference_type ROI2D::region_nlinfo::first_pos_idx() const { 
    #ifndef NDEBUG 
    chk_nonempty_op("first_pos_idx()"); 
    #endif 

    return left - left_nl; 
}

inline ROI2D::difference_type ROI2D::region_nlinfo::first_pos_p1() const { 
    #ifndef NDEBUG 
    chk_nonempty_op("first_pos_p1()"); 
    #endif 

    return nodelist(0,first_pos_idx()); 
}

inline ROI2D::difference_type ROI2D::region_nlinfo::first_pos_p2() const { 
    #ifndef NDEBUG 
    chk_nonempty_op("first_pos_p2()"); 
    #endif 

    return left; 
}

inline ROI2D::difference_type ROI2D::region_nlinfo::last_pos_idx() const { 
    #ifndef NDEBUG 
    chk_nonempty_op("last_pos_idx()"); 
    #endif 

    return nodelist.width() - (right_nl - right) - 1; 
}

inline ROI2D::difference_type ROI2D::region_nlinfo::last_pos_p1() const { 
    #ifndef NDEBUG 
    chk_nonempty_op("last_pos_p1()"); 
    #endif 

    return nodelist(noderange(last_pos_idx()) - 1, last_pos_idx()); 
}

inline ROI2D::difference_type ROI2D::region_nlinfo::last_pos_p2() const { 
    #ifndef NDEBUG 
    chk_nonempty_op("last_pos_p2()"); 
    #endif 

    return right; 
}

// Interface functions -------------------------------------------------------//
template <typename T, typename T_container>
T_container& fill(T_container &A, const ROI2D::region_nlinfo &nlinfo, const T &val) {
    typedef ROI2D::difference_type                              difference_type;
    
    for (difference_type nl_idx = 0; nl_idx < nlinfo.nodelist.width(); ++nl_idx) {
        difference_type p2 = nl_idx + nlinfo.left_nl;
        for (difference_type np_idx = 0; np_idx < nlinfo.noderange(nl_idx); np_idx += 2) {
            difference_type np_top = nlinfo.nodelist(np_idx, nl_idx);
            difference_type np_bottom = nlinfo.nodelist(np_idx + 1, nl_idx);
            for (difference_type p1 = np_top; p1 <= np_bottom; ++p1) {
                A(p1,p2) = val;
            }
        }
    }
    
    return A;    
}

template <typename T, typename T_container>
T_container& fill(T_container &A, const Array2D<double>&boundary, const T &val) {    
    typedef ROI2D::difference_type                              difference_type;
    
    if (boundary.width() != 2) {
        throw std::invalid_argument("Input boundary has size: " + boundary.size_2D_string() + ". Boundary must have a width of 2.");
    }
    
    // Flood fill algorithm from : http://alienryderflex.com/polygon_fill/
    // boundary must have width of 2, where the first column is p1 coordinates
    // and the second column is p2 coordinates.
    
    if (boundary.empty()) {
        // If boundary is empty just return        
        return A;
    }
        
    // node_buf is used to hold nodes that are calculated to paint the polygon
    // along the sweep line. It is initialized to hold at least the max number 
    // of nodes.
    Array2D<double> node_buf(std::max(boundary.height(),difference_type(2)),1);
    // Get bounds from max and min of p2 coordinates in boundary. Also note that
    // boundary is non-empty at this point so min and max exist.
    difference_type left = std::ceil(std::max(*std::min_element(boundary.get_pointer() + boundary.height() - 1, boundary.get_pointer() + 2 * boundary.height()), 0.0));
    difference_type right = std::floor(std::min(*std::max_element(boundary.get_pointer() + boundary.height() - 1, boundary.get_pointer() + 2 * boundary.height()), A.width()-1.0));
    for (difference_type p2_sweep = left; p2_sweep <= right; ++p2_sweep) {
        difference_type buf_length = 0; // Keeps track of # of nodes
        // This will cycle over each line segment (point0->point1) of the polygon 
        // and test for intersections with a vertical sweep line on integer pixel 
        // locations.
        for (difference_type point1_idx = 0, point0_idx = boundary.height() - 1; point1_idx < boundary.height(); point0_idx = point1_idx++) {
            double point0_p1 = boundary(point0_idx,0), point0_p2 = boundary(point0_idx,1);
            double point1_p1 = boundary(point1_idx,0), point1_p2 = boundary(point1_idx,1);
            if ((p2_sweep < point1_p2 && p2_sweep >= point0_p2) || 
                (p2_sweep < point0_p2 && p2_sweep >= point1_p2)) {        
                node_buf(buf_length++) = point1_p1 + (p2_sweep-point1_p2)/(point0_p2-point1_p2) * (point0_p1-point1_p1);
            } 
        }
                                        
        // Sort nodes
        std::sort(node_buf.get_pointer(), node_buf.get_pointer() + buf_length);
                
        // Paint nodes
        for (difference_type idx = 0; idx < buf_length; idx += 2) {
            difference_type np_top = std::ceil(node_buf(idx));
            difference_type np_bottom = std::floor(node_buf(idx+1));
            if (np_top >= A.height()) { 
                break; // top node is lower than bottom of the mask
            }
            if (np_bottom >= 0) { // bottom node is lower than the top of the mask
                if (np_top < 0) { np_top = 0; }
                if (np_bottom >= A.height()) { np_bottom = A.height() - 1; }
                
                // At this point the nodes are within the mask bounds, so paint 
                // them. Note that it's possible for np_top > np_bottom in the 
                // case that the boundary is between two pixels. In this case,
                // the loop does nothing, so its safe.
                for (difference_type p1 = np_top; p1 <= np_bottom; p1++) { 
                    A(p1,p2_sweep) = val;
                }
            }
        }
    }
    
    return A;    
}

template <typename T_container>
std::pair<typename T_container::value_type, typename T_container::coords> max(T_container &A, const ROI2D::region_nlinfo &nlinfo) {
    typedef ROI2D::difference_type                              difference_type;
    
    if (nlinfo.empty()) {
        throw std::invalid_argument("Attempted to find the max value in Array using an empty nlinfo.");
    }
    
    difference_type p1_max = nlinfo.first_pos_p1();
    difference_type p2_max = nlinfo.first_pos_p2();
    difference_type val_max = A(nlinfo.first_pos_idx());
    for (difference_type nl_idx = 0; nl_idx < nlinfo.nodelist.width(); ++nl_idx) {
        difference_type p2 = nl_idx + nlinfo.left_nl;
        for (difference_type np_idx = 0; np_idx < nlinfo.noderange(nl_idx); np_idx += 2) {
            difference_type np_top = nlinfo.nodelist(np_idx, nl_idx);
            difference_type np_bottom = nlinfo.nodelist(np_idx + 1, nl_idx);
            for (difference_type p1 = np_top; p1 <= np_bottom; ++p1) {
                if (A(p1,p2) > val_max) {
                    p1_max = p1;
                    p2_max = p2;
                    val_max = A(p1,p2);
                }
            }
        }
    }
    
    return { val_max,  { p1_max,p2_max } };    
}

template <typename T_container>
std::pair<typename T_container::value_type, typename T_container::coords> min(T_container &A, const ROI2D::region_nlinfo &nlinfo) {
    typedef ROI2D::difference_type                              difference_type;
    
    if (nlinfo.empty()) {
        throw std::invalid_argument("Attempted to find the min value in Array using an empty nlinfo.");
    }
    
    difference_type p1_min = nlinfo.first_pos_p1();
    difference_type p2_min = nlinfo.first_pos_p2();
    difference_type val_min = A(nlinfo.first_pos_idx());
    for (difference_type nl_idx = 0; nl_idx < nlinfo.nodelist.width(); ++nl_idx) {
        difference_type p2 = nl_idx + nlinfo.left_nl;
        for (difference_type np_idx = 0; np_idx < nlinfo.noderange(nl_idx); np_idx += 2) {
            difference_type np_top = nlinfo.nodelist(np_idx, nl_idx);
            difference_type np_bottom = nlinfo.nodelist(np_idx + 1, nl_idx);
            for (difference_type p1 = np_top; p1 <= np_bottom; ++p1) {
                if (A(p1,p2) < val_min) {
                    p1_min = p1;
                    p2_min = p2;
                    val_min = A(p1,p2);
                }
            }
        }
    }
    
    return { val_min,  { p1_min,p2_min } };    
}

}

#endif	/* ROI2D_H */