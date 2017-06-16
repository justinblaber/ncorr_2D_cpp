/* 
 * File:   ROI2D.cpp
 * Author: justin
 * 
 * Created on February 11, 2015, 12:02 AM
 */

#include "ROI2D.h"

namespace ncorr {  

// Additional Constructors ---------------------------------------------------//    
ROI2D::ROI2D(Array2D<bool> mask, difference_type cutoff) : mask_ptr(std::make_shared<Array2D<bool>>(std::move(mask))), regions_ptr(std::make_shared<std::vector<region>>()) {
    if (cutoff < 0) {        
        throw std::invalid_argument("Attempted to form ROI2D with cutoff of: " + std::to_string(cutoff) + 
                                    ". cutoff must be an integer of value 0 or greater.");
    }
    
    // Given an input of a mask, form ROI2D ----------------------------------//    
    // Form the nlinfos first
    auto nlinfos_pair = ROI2D::region_nlinfo::form_nlinfos(*this->mask_ptr, cutoff);
                
    // Cycle over each nlinfo and create a corresponding region_boundary
    Array2D<bool> mask_buf(this->mask_ptr->height(), this->mask_ptr->width());
    for (const auto &nlinfo : nlinfos_pair.first) {
        auto boundary = nlinfo.to_boundary(mask_buf);
        this->regions_ptr->emplace_back(std::move(nlinfo), std::move(boundary));          
    }
    
    if (nlinfos_pair.second) {
        // If regions were removed, update the mask
        draw_mask();
    }    
    
    // Set points
    set_points();    
}

ROI2D::ROI2D(region_nlinfo nlinfo, difference_type h, difference_type w) : mask_ptr(std::make_shared<Array2D<bool>>(h,w)), regions_ptr(std::make_shared<std::vector<region>>()) {
    // Given an input of an nlinfo, form ROI2D. .
                  
    // Set regions - can use mask_ptr as buffer since it's empty at this point
    auto boundary = nlinfo.to_boundary(*this->mask_ptr);
    this->regions_ptr->emplace_back(std::move(nlinfo), std::move(boundary));   
    
    // Draw mask
    draw_mask();
    
    // Set points
    set_points();
}

ROI2D::ROI2D(region_boundary boundary, difference_type h, difference_type w) : mask_ptr(std::make_shared<Array2D<bool>>(h,w)), regions_ptr(std::make_shared<std::vector<region>>()) {
    // Given an input of a boundary, form ROI2D.
       
    // Set region - can use mask_ptr as buffer since it's empty at this point
    auto nlinfo = boundary.to_nlinfo(*this->mask_ptr);
    this->regions_ptr->emplace_back(std::move(nlinfo), std::move(boundary));
    
    // Draw mask
    draw_mask();
    
    // Set points
    set_points();
}

ROI2D::ROI2D(std::vector<region_nlinfo> nlinfos, difference_type h, difference_type w) : mask_ptr(std::make_shared<Array2D<bool>>(h,w)), regions_ptr(std::make_shared<std::vector<region>>()) {
    // Given input of nlinfos, form ROI2D.
        
    // Set region(s) - can use mask_ptr as buffer since it's empty at this point
    for (const auto &nlinfo : nlinfos) {      
        auto boundary = nlinfo.to_boundary(*this->mask_ptr);
        this->regions_ptr->emplace_back(std::move(nlinfo), std::move(boundary));   
    }
    
    // Draw mask
    draw_mask();
    
    // Set points
    set_points();
}

ROI2D::ROI2D(std::vector<region_boundary> boundaries, difference_type h, difference_type w) : mask_ptr(std::make_shared<Array2D<bool>>(h,w)), regions_ptr(std::make_shared<std::vector<region>>()) {
    // Given an input of boundaries, form ROI2D.
        
    // Set region(s) - can use mask_ptr as buffer since it's empty at this point
    for (const auto &boundary : boundaries) {
        auto nlinfo = boundary.to_nlinfo(*this->mask_ptr);
        this->regions_ptr->emplace_back(std::move(nlinfo), std::move(boundary));
    }
    
    // Draw mask
    draw_mask();
    
    // Set points
    set_points();
}

// Static factory methods ----------------------------------------------------//

// Note that 'simple' ROIs require:
//  1) Contains 1 region/nlinfo
//  2) nlinfo must have two nodes per column
//  3) Mask must have odd width
//  4) Must contain its center point. 
ROI2D ROI2D::simple_circle(difference_type r) {      
    difference_type length = 2*r + 1;
    double r_squared = std::pow(r,2);  
    
    Array2D<bool> mask(length, length);
    for (difference_type p2 = 0; p2 < length; ++p2) {
        double h_squared = std::pow(p2-r,2);
        difference_type np_top = std::ceil(-std::sqrt(r_squared - h_squared)) + r;
        difference_type np_bottom = std::floor(std::sqrt(r_squared - h_squared)) + r;
        for (difference_type p1 = np_top; p1 <= np_bottom; ++p1) {
            mask(p1,p2) = true;
        }
    }
    
    return ROI2D(std::move(mask));
}

ROI2D ROI2D::simple_square(difference_type r) { 
    Array2D<bool> mask(2*r+1, 2*r+1, true);
    
    return ROI2D(std::move(mask));
}

ROI2D ROI2D::load(std::ifstream &is) {
    // Form empty ROI2D then fill in values in accordance to how they are saved
    ROI2D roi;
    
    // Load mask
    roi.mask_ptr = std::make_shared<Array2D<bool>>(Array2D<bool>::load(is));
    
    // Load regions
    difference_type num_regions = 0;
    is.read(reinterpret_cast<char*>(&num_regions), std::streamsize(sizeof(difference_type)));
    roi.regions_ptr = std::make_shared<std::vector<ROI2D::region>>(num_regions);
    for (auto &reg : *roi.regions_ptr) {
        reg = ROI2D::region::load(is);
    }
    
    // Load points
    is.read(reinterpret_cast<char*>(&roi.points), std::streamsize(sizeof(difference_type)));
    
    return roi;
}

// Operators Interface -------------------------------------------------------//  
std::ostream& operator<<(std::ostream &os, const ROI2D &roi) {
    os << "Mask :" << '\n';
    os << *roi.mask_ptr;             
    
    for (const auto &region : *roi.regions_ptr) {
        os << region << '\n';
    }
    
    os << "Total points: " << roi.points;
    
    return os;
}

bool isequal(const ROI2D &roi1, const ROI2D &roi2) {
    typedef ROI2D::difference_type                              difference_type;
    
    if (roi1.points == roi2.points && isequal(*roi1.mask_ptr, *roi2.mask_ptr) && roi1.regions_ptr->size() == roi2.regions_ptr->size()) {
        for (difference_type region_idx = 0; region_idx < difference_type(roi1.regions_ptr->size()); ++region_idx) {
            if (!isequal((*roi1.regions_ptr)[region_idx], (*roi2.regions_ptr)[region_idx])) {
                return false;
            }
        }
        
        // At this point, the points, masks, and all the regions in roi1 and roi2
        // are the same
        return true;
    }
    
    return false;    
}

void save(const ROI2D &roi, std::ofstream &os) {    
    typedef ROI2D::difference_type                              difference_type;
    
    // Save mask -> regions -> points
    save(*roi.mask_ptr, os);
    
    difference_type num_regions = roi.regions_ptr->size();
    os.write(reinterpret_cast<const char*>(&num_regions), std::streamsize(sizeof(difference_type)));
    for (const auto &reg : *roi.regions_ptr) {
        save(reg, os);
    }
    
    os.write(reinterpret_cast<const char*>(&roi.points), std::streamsize(sizeof(difference_type)));
}

// Access --------------------------------------------------------------------//
const ROI2D::region_nlinfo& ROI2D::get_nlinfo(difference_type region_idx) const { 
    chk_region_idx_in_bounds(region_idx, "get_nlinfo()");
    
    return (*regions_ptr)[region_idx].nlinfo; 
}

const ROI2D::region_boundary& ROI2D::get_boundary(difference_type region_idx) const { 
    chk_region_idx_in_bounds(region_idx, "get_boundary()");
    
    return (*regions_ptr)[region_idx].boundary; 
}

std::pair<ROI2D::difference_type,ROI2D::difference_type> ROI2D::get_region_idx(difference_type p1, difference_type p2) const {
   // .first refers to idx of region - will be -1 if it does not exist
   // .second refers to idx of nodepair which contains p2 - will be -1 if it does not exists
   if (!mask_ptr->in_bounds(p1,p2)) {
       return {-1,-1};
   } 
   
   for (difference_type region_idx = 0; region_idx < difference_type(regions_ptr->size()); ++region_idx) {
        const auto &nlinfo = (*regions_ptr)[region_idx].nlinfo;
        difference_type nl_idx = p2 - nlinfo.left_nl;
        if (nlinfo.noderange.in_bounds(nl_idx)) {
            for (difference_type np_idx = 0; np_idx < nlinfo.noderange(nl_idx); np_idx += 2) {
                difference_type np_top = nlinfo.nodelist(np_idx, nl_idx);      
                difference_type np_bottom = nlinfo.nodelist(np_idx + 1, nl_idx);                  
                if (p1 < np_top) {
                    break; // p1 comes before top of nodepair
                }               
                if (p1 <= np_bottom) {
                    return {region_idx, np_idx}; // p1 is contained in this nodepair
                }
            }
        }
   }  
   // No nlinfo contained p1 and p2
   return {-1,-1};
}

// Arithmetic operations ------------------------------------------------------//
ROI2D ROI2D::reduce(difference_type scalefactor) const {
    if (scalefactor < 1) {
        throw std::invalid_argument("Attempted to reduce ROI2D with scalefactor of: " + std::to_string(scalefactor) + 
                                    ". scalefactor must be an integer of value 1 or greater.");
    }
    
    // Reduction is done by reducing nlinfos. It is possible for nlinfo to 
    // disappear after reduction, but an empty nlinfo is kept in order to 
    // maintain a 1 to 1 correspondence of regions. If nlinfo becomes 
    // non-contiguous after reduction, reduce() returns the largest contiguous 
    // region.
    std::vector<region_nlinfo> nlinfos_reduced;
    Array2D<bool> mask_buf(height(), width());
    for (difference_type region_idx = 0; region_idx < difference_type(regions_ptr->size()); ++region_idx) {
        nlinfos_reduced.push_back((*regions_ptr)[region_idx].nlinfo.reduce(scalefactor, mask_buf));
    }
    
    return ROI2D(std::move(nlinfos_reduced), std::ceil(double(height())/scalefactor), std::ceil(double(width())/scalefactor));
}

ROI2D ROI2D::form_union(const Array2D<bool> &mask) const {
    if (mask.height() != height() || mask.width() != width()) {
        throw std::invalid_argument("Attempted to form union of ROI2D of size: " + size_2D_string() + 
                                    " with mask of size : " + mask.size_2D_string() + ". Mask and ROI2D must have the same size.");
    }
    
    // Union is done by finding new nodepair values depending on the values of 
    // the mask between the original nodepair values. It is possible for nlinfo to 
    // disappear after union, but an empty nlinfo is kept in order to maintain a 
    // 1 to 1 correspondence of regions. If nlinfo becomes non-contiguous after 
    // union, form_union() returns the largest contiguous region.
    std::vector<region_nlinfo> nlinfos_unioned;
    Array2D<bool> mask_buf(height(), width());
    for (difference_type region_idx = 0; region_idx < difference_type(regions_ptr->size()); ++region_idx) {
        nlinfos_unioned.push_back((*regions_ptr)[region_idx].nlinfo.form_union(mask,mask_buf));
    }
    
    return ROI2D(std::move(nlinfos_unioned), height(), width());
}   

// Incrementor ---------------------------------------------------------------//
ROI2D::incrementor ROI2D::begin_inc() const { 
    return (regions_ptr->empty() ? incrementor(*this, 0, { }) : incrementor(*this, 0, regions_ptr->front().nlinfo.begin_inc())); 
}

ROI2D::incrementor ROI2D::end_inc() const { 
    return (regions_ptr->empty() ? incrementor(*this, 0, { }) : incrementor(*this, regions_ptr->size(), regions_ptr->back().nlinfo.end_inc())); 
}

// contig_subregion_generator ------------------------------------------------//
ROI2D::contig_subregion_generator ROI2D::get_contig_subregion_generator(SUBREGION subregion_type, difference_type r) const { 
    return contig_subregion_generator(*this, subregion_type, r);
}

// Utility methods -----------------------------------------------------------//
void ROI2D::set_points() {
    points = 0;
    for (const auto &region : *regions_ptr) {
        points += region.nlinfo.points;
    }
}

void ROI2D::draw_mask() {
    (*this->mask_ptr)() = false;
    for (const auto &region : *regions_ptr) {
        fill(*this->mask_ptr, region.nlinfo, true);
    }
}

// Checks --------------------------------------------------------------------//
void ROI2D::chk_region_idx_in_bounds(difference_type idx, const std::string &op) const {
    if (idx < 0 || idx >= difference_type(regions_ptr->size())) {
        throw std::invalid_argument("Attempted to access region " + std::to_string(idx) + " for " + op +  " operation, but there are only " + std::to_string(regions_ptr->size()) + " regions.");
    }    
}

// Static factory methods ----------------------------------------------------//
ROI2D::region_nlinfo ROI2D::region_nlinfo::load(std::ifstream &is) {
    // Form empty nlinfo then fill in values in accordance to how they are saved
    region_nlinfo nlinfo;
    
    // Load bounds
    is.read(reinterpret_cast<char*>(&nlinfo.top), std::streamsize(sizeof(difference_type)));
    is.read(reinterpret_cast<char*>(&nlinfo.bottom), std::streamsize(sizeof(difference_type)));
    is.read(reinterpret_cast<char*>(&nlinfo.left), std::streamsize(sizeof(difference_type)));
    is.read(reinterpret_cast<char*>(&nlinfo.right), std::streamsize(sizeof(difference_type)));
    is.read(reinterpret_cast<char*>(&nlinfo.left_nl), std::streamsize(sizeof(difference_type)));
    is.read(reinterpret_cast<char*>(&nlinfo.right_nl), std::streamsize(sizeof(difference_type)));
    
    // Load nodelist and noderange
    nlinfo.nodelist = Array2D<difference_type>::load(is);
    nlinfo.noderange = Array2D<difference_type>::load(is);
    
    // Load points
    is.read(reinterpret_cast<char*>(&nlinfo.points), std::streamsize(sizeof(difference_type)));
    
    return nlinfo;
}

// Operations interface ------------------------------------------------------//
std::ostream& operator<<(std::ostream &os, const ROI2D::region_nlinfo &nlinfo) {        
    os << "nlinfo: " << '\n';
    os << "Top bound of nlinfo: " << std::to_string(nlinfo.top) << "." << '\n';
    os << "Bottom bound of nlinfo: " << std::to_string(nlinfo.bottom) << "." << '\n';
    os << "Left bound of nlinfo: " << std::to_string(nlinfo.left) << "." << '\n';
    os << "Right bound of nlinfo: " << std::to_string(nlinfo.right) << "." << '\n';
    os << "Left position of nodelist: " << std::to_string(nlinfo.left_nl) << "." << '\n';
    os << "Right position of nodelist: " << std::to_string(nlinfo.right_nl) << "." << '\n';
        
    os << '\n' << "nodelist: " << '\n';
    os << nlinfo.nodelist << '\n';
    os << "noderange: " << '\n';
    os << nlinfo.noderange << '\n';
    
    os << '\n' << "nlinfo points: " << std::to_string(nlinfo.points) << "." << '\n';
    
    return os;
}  

bool isequal(const ROI2D::region_nlinfo &nlinfo1, const ROI2D::region_nlinfo &nlinfo2) {
    return nlinfo1.top == nlinfo2.top &&
           nlinfo1.bottom == nlinfo2.bottom &&
           nlinfo1.left == nlinfo2.left &&
           nlinfo1.right == nlinfo2.right &&
           nlinfo1.left_nl == nlinfo2.left_nl &&
           nlinfo1.right_nl == nlinfo2.right_nl &&
           isequal(nlinfo1.nodelist,nlinfo2.nodelist) &&
           isequal(nlinfo1.noderange,nlinfo2.noderange) &&
           nlinfo1.points == nlinfo2.points;
}

void save(const ROI2D::region_nlinfo &nlinfo, std::ofstream &os) {    
    typedef ROI2D::difference_type                              difference_type;
    
    // Save bounds -> nodelist -> noderange -> points
    os.write(reinterpret_cast<const char*>(&nlinfo.top), std::streamsize(sizeof(difference_type)));
    os.write(reinterpret_cast<const char*>(&nlinfo.bottom), std::streamsize(sizeof(difference_type)));
    os.write(reinterpret_cast<const char*>(&nlinfo.left), std::streamsize(sizeof(difference_type)));
    os.write(reinterpret_cast<const char*>(&nlinfo.right), std::streamsize(sizeof(difference_type)));
    os.write(reinterpret_cast<const char*>(&nlinfo.left_nl), std::streamsize(sizeof(difference_type)));
    os.write(reinterpret_cast<const char*>(&nlinfo.right_nl), std::streamsize(sizeof(difference_type)));
        
    save(nlinfo.nodelist, os);
    save(nlinfo.noderange, os);
    
    os.write(reinterpret_cast<const char*>(&nlinfo.points), std::streamsize(sizeof(difference_type)));
}

// Arithmetic methods --------------------------------------------------------//
bool ROI2D::region_nlinfo::in_nlinfo(difference_type p1, difference_type p2) const {
    difference_type nl_idx = p2 - left_nl;
    if (noderange.in_bounds(nl_idx)) {
        for (difference_type np_idx = 0; np_idx < noderange(nl_idx); np_idx += 2) {
            difference_type np_top = nodelist(np_idx, nl_idx);      
            difference_type np_bottom = nodelist(np_idx + 1, nl_idx);                  
            if (p1 < np_top) {
                return false; // p1 comes before top of nodepair
            }               
            if (p1 <= np_bottom) {
                return true;  // p1 is contained in this nodepair
            }
        }
    }
    
    return false;
}

ROI2D::region_nlinfo& ROI2D::region_nlinfo::shift(difference_type p1_shift, difference_type p2_shift) {
    // Note that no checking is done for shifting out of bounds so be careful.
    
    // Shift bounds first    
    top += p1_shift;
    bottom += p1_shift;
    left += p2_shift;
    right += p2_shift;
    left_nl += p2_shift;
    right_nl += p2_shift;
    
    // Shift nodelist
    for (difference_type nl_idx = 0; nl_idx < nodelist.width(); ++nl_idx) {
        for (difference_type np_idx = 0; np_idx < noderange(nl_idx); ++np_idx) {
            nodelist(np_idx,nl_idx) += p1_shift;
        }
    }
    
    return *this;
}

// Incrementor ---------------------------------------------------------------//
ROI2D::region_nlinfo::incrementor ROI2D::region_nlinfo::begin_inc() const { 
    return (empty() ? incrementor(*this, 0, 0, 0) : incrementor(*this, first_pos_idx(), 0, first_pos_p1())); 
}

ROI2D::region_nlinfo::incrementor ROI2D::region_nlinfo::end_inc() const { 
    return (empty() ? incrementor(*this, 0, 0, 0) : incrementor(*this, nodelist.width(), noderange(last_pos_idx()), last_pos_p1() + 1)); 
}

// Static factory methods ----------------------------------------------------//
namespace details {        
    void add_interacting_nodes(ROI2D::difference_type np_adj_p2, 
                               ROI2D::difference_type np_loaded_top,
                               ROI2D::difference_type np_loaded_bottom,
                               Array2D<std::vector<ROI2D::difference_type>> &overall_nodelist,
                               Array2D<std::vector<bool>> &overall_active_nodepairs,
                               std::stack<ROI2D::difference_type> &queue_np_idx) {
        typedef ROI2D::difference_type                          difference_type;
        
        // Make sure adjacent nodepair(s) position is in range.
        if (overall_nodelist.in_bounds(np_adj_p2)) {
            // Scans nodes from top to bottom
            for (difference_type np_adj_idx = 0; np_adj_idx < difference_type(overall_nodelist(np_adj_p2).size()); np_adj_idx += 2) {
                difference_type np_adj_top = overall_nodelist(np_adj_p2)[np_adj_idx];
                difference_type np_adj_bottom = overall_nodelist(np_adj_p2)[np_adj_idx + 1];
                if (np_loaded_bottom < np_adj_top) {
                    return; // top node of adjacent nodepair is below bottom node of loaded nodepair
                }                 
                if (overall_active_nodepairs(np_adj_p2)[np_adj_idx/2] && np_loaded_top <= np_adj_bottom) {
                    // Inactivate node pair, and then insert into the queue
                    overall_active_nodepairs(np_adj_p2)[np_adj_idx/2] = false;
                    queue_np_idx.push(np_adj_top);
                    queue_np_idx.push(np_adj_bottom);
                    queue_np_idx.push(np_adj_p2);
                } 
            }
        }
    }
}

std::pair<std::vector<ROI2D::region_nlinfo>,bool> ROI2D::region_nlinfo::form_nlinfos(const Array2D<bool> &mask, difference_type cutoff) {            
    if (cutoff < 0) {        
        throw std::invalid_argument("Attempted to form nlinfos with cutoff of: " + std::to_string(cutoff) + 
                                    ". cutoff must be an integer of value 0 or greater.");
    }

    // Form overall_nodelist and overall_active_nodepairs --------------------//
    Array2D<std::vector<difference_type>> overall_nodelist(1,mask.width());
    Array2D<std::vector<bool>> overall_active_nodepairs(1,mask.width());
    for (difference_type p2 = 0; p2 < mask.width(); ++p2) {
        bool in_nodepair = false;
        for (difference_type p1 = 0; p1 < mask.height(); ++p1) { 
            if (!in_nodepair && mask(p1,p2)) {
                in_nodepair = true;
                overall_nodelist(p2).push_back(p1); // sets top node
            }
            if (in_nodepair && (!mask(p1,p2) || p1 == mask.height()-1)) {
                in_nodepair = false;
                overall_nodelist(p2).push_back((p1 == mask.height()-1 && mask(p1,p2)) ? p1 : p1-1); // Sets bottom node
            }
        }

        // Update overall_active_nodepairs - this keeps track of which node
        // pairs have been analyzed when doing contiguous separation; set to 
        // true initially.
        overall_active_nodepairs(p2).resize(overall_nodelist(p2).size()/2, true);
    }

    // Separate regions ------------------------------------------------------//   
    // Regions are made 4-way contiguous. Scan over columns and separate 
    // contiguous nodelists in overall_nodelist.
    std::vector<ROI2D::region_nlinfo> nlinfos; // This will get updated and returned
    bool removed = false;                      // Keeps track if regions are removed due to "cutoff" parameter
    for (difference_type p2_sweep = 0; p2_sweep < overall_nodelist.width(); ++p2_sweep) {
        // Find first active node pair
        difference_type np_init_idx = -1;
        for (difference_type np_idx = 0; np_idx < difference_type(overall_nodelist(p2_sweep).size()); np_idx += 2) {
            if (overall_active_nodepairs(p2_sweep)[np_idx/2]) {       // Test if nodepair is active
                overall_active_nodepairs(p2_sweep)[np_idx/2] = false; // Inactivate node pair
                np_init_idx = np_idx;                                 // Store nodepair idx
                break;
            }
        }

        // If there are no active node pairs, then continue to next column
        if (np_init_idx == -1) {
            continue;
        }

        // nlinfo_buf to be updated, then inserted into nlinfos
        ROI2D::region_nlinfo nlinfo_buf(mask.height()-1,    // Top bound of region                (this gets updated)
                                        0,                  // Bottom bound of region             (this gets updated)
                                        p2_sweep,           // Left bound of region               (this is correct)
                                        0,                  // Right bound of region              (this gets updated)
                                        p2_sweep,           // Left position of nodelist          (this is correct)
                                        0,                  // Right position of nodelist         (this gets updated)
                                        0,                  // h of nodelist                      (this gets updated)  
                                        0,                  // w of nodelist                      (this gets updated)     
                                        0);                 // Number of points in region         (this gets updated) 

        // Keep track of nodes with separate_nodelist
        Array2D<std::vector<difference_type>> separate_nodelist(1,mask.width());

        // Initialize queue and enter while loop - exit when queue is empty
        std::stack<difference_type> queue_np_idx;                       // Holds all nodepairs (along with their index) which need to be processed
        queue_np_idx.push(overall_nodelist(p2_sweep)[np_init_idx]);     // Top
        queue_np_idx.push(overall_nodelist(p2_sweep)[np_init_idx + 1]); // Bottom
        queue_np_idx.push(p2_sweep);                                    // position of nodepair
        while (!queue_np_idx.empty()) {
            // Pop nodepair and its position out of queue and compare 
            // it to adjacent nodepairs (left and right of np_loaded_p2)
            difference_type np_loaded_p2 = queue_np_idx.top(); queue_np_idx.pop();
            difference_type np_loaded_bottom = queue_np_idx.top(); queue_np_idx.pop();
            difference_type np_loaded_top = queue_np_idx.top(); queue_np_idx.pop();

            // Compare to node pairs LEFT. Any node pairs which interact are added to the queue
            details::add_interacting_nodes(np_loaded_p2 - 1, 
                                           np_loaded_top, 
                                           np_loaded_bottom,
                                           overall_nodelist, 
                                           overall_active_nodepairs, 
                                           queue_np_idx);

            // Compare to node pairs RIGHT. Any node pairs which interact are added to the queue
            details::add_interacting_nodes(np_loaded_p2 + 1, 
                                           np_loaded_top, 
                                           np_loaded_bottom, 
                                           overall_nodelist, 
                                           overall_active_nodepairs, 
                                           queue_np_idx);

            // Update points
            nlinfo_buf.points += np_loaded_bottom - np_loaded_top + 1;

            // Update bounds - note that "right_nl" and "right" are the same
            if (np_loaded_top < nlinfo_buf.top) { nlinfo_buf.top = np_loaded_top; }                         // Top
            if (np_loaded_bottom > nlinfo_buf.bottom) { nlinfo_buf.bottom = np_loaded_bottom; }             // Bottom
            if (np_loaded_p2 > nlinfo_buf.right) { nlinfo_buf.right_nl = nlinfo_buf.right = np_loaded_p2; } // Right

            // Insert node pairs and then sort - usually very small so BST isn't
            // necessary.
            separate_nodelist(np_loaded_p2).push_back(np_loaded_top);
            separate_nodelist(np_loaded_p2).push_back(np_loaded_bottom);
            std::sort(separate_nodelist(np_loaded_p2).begin(), separate_nodelist(np_loaded_p2).end());
        }

        // Now finish setting nodelist and noderange for this region.
        // Find max nodes first so we can use it to set the correct height
        // for nodelist.
        difference_type max_nodes = 0;
        for (const auto &nodes : separate_nodelist) {
            if (difference_type(nodes.size()) > max_nodes) {
                max_nodes = nodes.size();
            }
        }

        // Set and fill nodelist and noderange
        nlinfo_buf.nodelist = Array2D<difference_type>(max_nodes, nlinfo_buf.right_nl - nlinfo_buf.left_nl + 1);
        nlinfo_buf.noderange = Array2D<difference_type>(1, nlinfo_buf.right_nl - nlinfo_buf.left_nl + 1);
        for (difference_type nl_idx = 0; nl_idx < nlinfo_buf.nodelist.width(); ++nl_idx) {
            difference_type p2 = nl_idx + nlinfo_buf.left_nl;
            // noderange:
            nlinfo_buf.noderange(nl_idx) = separate_nodelist(p2).size();
            // nodelist:
            for (difference_type np_idx = 0; np_idx < difference_type(separate_nodelist(p2).size()); ++np_idx) {
                nlinfo_buf.nodelist(np_idx,nl_idx) = separate_nodelist(p2)[np_idx];
            }
        }

        // Subtract one from p2_sweep in order to recheck the column to 
        // ensure all nodes are deactivated before proceeding
        --p2_sweep;

        // Make sure number of points in nlinfo is greater than or equal to
        // the cutoff
        if (nlinfo_buf.points >= cutoff) {      
            nlinfos.push_back(nlinfo_buf);
        } else {
            removed = true; // Parameter lets caller know regions were removed
        }
    }

    return {std::move(nlinfos), removed};
}

// Arithmetic methods --------------------------------------------------------//
ROI2D::region_nlinfo ROI2D::region_nlinfo::reduce(difference_type scalefactor, Array2D<bool> &mask_buf) const {
    if (scalefactor < 1) {
        throw std::invalid_argument("Attempted to reduce nlinfo with scalefactor of: " + std::to_string(scalefactor) + 
                                    ". scalefactor must be an integer of value 1 or greater.");
    }
    
    // This function will return a copy of a reduced nlinfo based on input
    // scalefactor.
    
    if (empty()) {
        // Return empty nlinfo
        return region_nlinfo();      
    }
    
    // Only *_nl bounds can be set now. Other bounds must be calculated since 
    // regions can disappear if a large scalefactor is used. Note that since
    // nlinfo is nonempty at this point, using the bounds is valid.
    difference_type top_reduced = std::ceil(double(top)/scalefactor);
    difference_type bottom_reduced = std::floor(double(bottom)/scalefactor);
    difference_type left_reduced_nl = std::ceil(double(left_nl)/scalefactor);
    difference_type right_reduced_nl = std::floor(double(right_nl)/scalefactor);   
    region_nlinfo nlinfo_reduced(bottom_reduced,                              // top               (gets updated)
                                 top_reduced,                                 // bottom            (gets updated)
                                 right_reduced_nl,                            // left              (gets updated)
                                 left_reduced_nl,                             // right             (gets updated)
                                 left_reduced_nl,                             // left_nl           (correct)
                                 right_reduced_nl,                            // right_nl          (correct)
                                 nodelist.height(),                           // nodelist height   (correct)
                                 right_reduced_nl - left_reduced_nl + 1,      // nodelist width    (correct)
                                 0);                                          // points            (gets updated)
    
    for (difference_type nl_idx_reduced = 0; nl_idx_reduced < nlinfo_reduced.nodelist.width(); ++nl_idx_reduced) {
        difference_type nl_idx = (nl_idx_reduced + nlinfo_reduced.left_nl) * scalefactor - left_nl;
        for (difference_type np_idx = 0; np_idx < noderange(nl_idx); np_idx += 2) {            
            difference_type np_top_reduced = std::ceil(double(nodelist(np_idx, nl_idx))/scalefactor);
            difference_type np_bottom_reduced = std::floor(double(nodelist(np_idx + 1, nl_idx))/scalefactor);
            if (np_bottom_reduced >= np_top_reduced) { 
                // Update points
                nlinfo_reduced.points += np_bottom_reduced - np_top_reduced + 1;
                
                // Update bounds
                if (np_top_reduced < nlinfo_reduced.top) { nlinfo_reduced.top = np_top_reduced; }
                if (np_bottom_reduced > nlinfo_reduced.bottom) { nlinfo_reduced.bottom = np_bottom_reduced; }
                if (nl_idx_reduced + nlinfo_reduced.left_nl < nlinfo_reduced.left) { nlinfo_reduced.left = nl_idx_reduced + nlinfo_reduced.left_nl; }
                if (nl_idx_reduced + nlinfo_reduced.left_nl > nlinfo_reduced.right) { nlinfo_reduced.right = nl_idx_reduced + nlinfo_reduced.left_nl; }
                
                // Insert nodepairs
                nlinfo_reduced.nodelist(nlinfo_reduced.noderange(nl_idx_reduced), nl_idx_reduced) = np_top_reduced;
                nlinfo_reduced.nodelist(nlinfo_reduced.noderange(nl_idx_reduced)+1, nl_idx_reduced) = np_bottom_reduced;
                
                // Update noderange
                nlinfo_reduced.noderange(nl_idx_reduced) += 2;
            }
        }
    }
    // Return largest contiguous region since reduced nlinfo may no longer be 
    // 4-way contiguous
    return nlinfo_reduced.largest_contig_nlinfo(mask_buf);
}

ROI2D::region_nlinfo ROI2D::region_nlinfo::form_union(const Array2D<bool> &mask, Array2D<bool> &mask_buf) const {   
    if (!mask.same_size(mask_buf)) {
        throw std::invalid_argument("Attempted to form union with mask of size: " + mask.size_2D_string() + 
                                    " with mask_buf of size : " + mask_buf.size_2D_string() + ". Mask and mask_buf must have the same size.");
    }
    
    // This function will return a copy of a unioned nlinfo based on input mask.
    
    if (empty()) {
        // Return empty nlinfo
        return region_nlinfo();      
    }
    
    // Only *_nl bounds can be set now. Other bounds must be calculated since 
    // it is indeterminate how many new nodes will be added.
    region_nlinfo nlinfo_union(bottom,           // top               (gets updated)
                               top,              // bottom            (gets updated)
                               right_nl,         // left              (gets updated)
                               left_nl,          // right             (gets updated)
                               left_nl,          // left_nl           (correct)
                               right_nl,         // right_nl          (correct)
                               0,                // nodelist height   (gets updated)
                               nodelist.width(), // nodelist width    (correct)
                               0);               // points            (gets updated)

    
    // Use a vector buffer to hold nodepairs
    Array2D<std::vector<difference_type>> nodelist_buf_vec(nodelist.width(),1);
    difference_type max_height = 0; // will be used to allocate array for nodelist

    // Cycle over nlinfo and find new nodepairs
    for (difference_type nl_idx = 0; nl_idx < nodelist.width(); ++nl_idx) {
        difference_type p2 = nl_idx + left_nl;
        for (difference_type np_idx = 0; np_idx < noderange(nl_idx); np_idx += 2) {
            difference_type np_top = nodelist(np_idx,nl_idx);
            difference_type np_bottom = nodelist(np_idx+1,nl_idx);
            bool in_nodepair = false;
            for (difference_type p1 = np_top; p1 <= np_bottom; p1++) {
                difference_type np_top_new, np_bottom_new;
                if (!in_nodepair && mask(p1,p2)) {
                    in_nodepair = true;
                    np_top_new = p1;
                    nodelist_buf_vec(nl_idx).push_back(np_top_new);
                }
                if (in_nodepair && (!mask(p1,p2) || p1 == np_bottom)) {
                    in_nodepair = false;
                    np_bottom_new = ((p1 == np_bottom && mask(p1,p2)) ? p1 : p1-1);

                    // update points
                    nlinfo_union.points += np_bottom_new - np_top_new + 1;

                    // update bounds
                    if (np_top_new < nlinfo_union.top) { nlinfo_union.top = np_top_new; }
                    if (np_bottom_new > nlinfo_union.bottom) { nlinfo_union.bottom = np_bottom_new; }
                    if (nl_idx + nlinfo_union.left_nl < nlinfo_union.left) { nlinfo_union.left = nl_idx + nlinfo_union.left_nl; }
                    if (nl_idx + nlinfo_union.left_nl > nlinfo_union.right) { nlinfo_union.right = nl_idx + nlinfo_union.left_nl; }

                    // insert bottom of nodepair
                    nodelist_buf_vec(nl_idx).push_back(np_bottom_new);

                    // Update noderange
                    nlinfo_union.noderange(nl_idx) += 2;

                    // Update max height
                    if (nlinfo_union.noderange(nl_idx) > max_height) { max_height = nlinfo_union.noderange(nl_idx); }
                }
            }
        }                
    }

    // Store nodepairs in nodelist_buf_vec in nlinfo_union
    nlinfo_union.nodelist = Array2D<difference_type>(max_height, nlinfo_union.nodelist.width());
    for (difference_type nl_idx = 0; nl_idx < nodelist.width(); ++nl_idx) {
        for (difference_type np_idx = 0; np_idx < difference_type(nodelist_buf_vec(nl_idx).size()); ++np_idx) {
            nlinfo_union.nodelist(np_idx,nl_idx) = nodelist_buf_vec(nl_idx)[np_idx];
        }
    }        
    // Return largest contiguous region since unioned nlinfo may no longer be 
    // 4-way contiguous
    return nlinfo_union.largest_contig_nlinfo(mask_buf);
}

namespace details {
    Array2D<double> calc_boundary(const Array2D<bool> &mask, double init_p1, double init_p2, ROI2D::difference_type init_direc = 0) {
        typedef ROI2D::difference_type                          difference_type;
        
        if (init_direc < 0 || init_direc > 3) {
            throw std::invalid_argument("Initial direction of: " + std::to_string(init_direc) + " was provided to calc_boundary function. Initial direction must be between (inclusive) 0 and 3.");
        }
        
        // Note that the returned boundary is "open" (does not contain duplicate 
        // initial and end points) and is the "outer boundary" of the mask (i.e. 
        // if the mask contains 1 true pixel, a boundary containing the 4 points 
        // of the corners of the pixel is returned). This will return a boundary 
        // for an 8-way connected region. init_p1 and init_p2 need to be the 
        // location WRT the outer boundary (i.e (1.5, 2.5) for a pixel located 
        // at (2, 3) assuming its the left-most top pixel). If this function is
        // called correctly, boundary is guaranteed to contain at least 4 points.
        
        // direc is based on:
        //
        //        1
        //        |
        //   2 ------- 0   
        //        |
        //        3
        //
        // Initial direction of 0 is correct for inputs containing the left-most
        // top pixel in an 8-way connected region.
        
        // Initialize p1, p2, and direc, which keep track of the current point 
        // location and the direction taken to get there, respectively.
        double p1 = init_p1;    
        double p2 = init_p2;
        difference_type direc = init_direc;
        
        // Initialize boundary
        std::vector<double> boundary_vec = {p1,p2};
        bool initpoint_encountered = false;
        while (!initpoint_encountered) {             
            // Get initial new direction
            direc = (direc + 3) % 4;

            // Cycle clockwise to find new direction
            for (difference_type it = 0; it < 4; ++it) {
                // Increment point based on direc
                double p1_inc, p2_inc;
                difference_type p1_inc_mask, p2_inc_mask;
                switch (direc) {
                    case 0: p1_inc = p1; p2_inc = p2 + 1; 
                            p1_inc_mask = std::round(p1_inc - 0.5); p2_inc_mask = std::round(p2_inc - 0.5); 
                            break;
                    case 1: p1_inc = p1 - 1; p2_inc = p2;
                            p1_inc_mask = std::round(p1_inc + 0.5); p2_inc_mask = std::round(p2_inc - 0.5); 
                            break;
                    case 2: p1_inc = p1; p2_inc = p2 - 1;
                            p1_inc_mask = std::round(p1_inc + 0.5); p2_inc_mask = std::round(p2_inc + 0.5); 
                            break;
                    case 3: p1_inc = p1 + 1; p2_inc = p2;
                            p1_inc_mask = std::round(p1_inc - 0.5); p2_inc_mask = std::round(p2_inc + 0.5); 
                            break;
                }            

                if (mask.in_bounds(p1_inc_mask,p2_inc_mask) && mask(p1_inc_mask,p2_inc_mask)) {
                    // This is the next point in the boundary
                    if (boundary_vec[0] != p1_inc || boundary_vec[1] != p2_inc) {
                        // Insert point into boundary and then set current point.
                        boundary_vec.push_back(p1_inc); boundary_vec.push_back(p2_inc);
                        p1 = p1_inc; p2 = p2_inc;
                    } else {
                        initpoint_encountered = true;
                    }              
                    // break from for loop
                    break; 
                } else {
                    // Increment clockwise one position, then re-check
                    direc = (direc + 1) % 4;
                }   
                
                if (it == 3) {
                    // This means an empty mask was provided and is a programmer 
                    // error
                    throw std::invalid_argument("Invalid initial point or empty mask was provided to calc_boundary() function.");
                }
            } 
        }    

        // Convert vec_boundary to Array2D
        Array2D<double> boundary(boundary_vec.size()/2, 2);
        for (difference_type idx = 0; idx < difference_type(boundary_vec.size()); idx += 2) {
            boundary(idx/2,0) = boundary_vec[idx];
            boundary(idx/2,1) = boundary_vec[idx+1];
        }
        
        return boundary;
    } 
}

ROI2D::region_nlinfo ROI2D::region_nlinfo::largest_contig_nlinfo(Array2D<bool> &mask_buf) const {
    if (empty()) {
        // nlinfo is empty, so return empty nlinfo.
        return ROI2D::region_nlinfo();
    }
            
    // Clear mask buffer and then fill with nlinfo
    mask_buf() = false;
    fill(mask_buf, *this, true);
    
    // Form new nlinfos and then return the one with the largest number of points
    auto nlinfos_pair = ROI2D::region_nlinfo::form_nlinfos(mask_buf);
    // nlinfo is non-empty at this point, so max element is guaranteed to exist.
    return *std::max_element(nlinfos_pair.first.begin(), nlinfos_pair.first.end(), [](const ROI2D::region_nlinfo &a, const ROI2D::region_nlinfo &b) { return a.points < b.points; });
}

ROI2D::region_boundary ROI2D::region_nlinfo::to_boundary(Array2D<bool> &mask_buf) const {
    // Converts input nlinfo to a region_boundary. Note that no bounds checking 
    // nor contiguity checking is done for input nlinfo's, so they must be 
    // correct.        

    if (empty()) {
        // If nlinfo is empty, return an empty boundary.
        return ROI2D::region_boundary();
    }

    // Form 1 "add" boundary and varying "sub" boundaries per nlinfo.
    // Create mask_buf and then fill nlinfo.
    mask_buf() = false;  
    fill(mask_buf, *this, true);

    // Get the "add" boundary ------------------------------------------------//
    // Use the left-most top pixel in the region; must use the "outer" position, 
    // so subtract by 0.5.
    auto add_boundary = details::calc_boundary(mask_buf, first_pos_p1() - 0.5, first_pos_p2() - 0.5);

    // Find the "sub" boundaries ---------------------------------------------//  
    // Use mask_inv_buf to keep track of which sub boundaries have been analyzed.
    Array2D<bool> mask_inv_buf(mask_buf.height(), mask_buf.width());  
    fill(mask_inv_buf, add_boundary, true);                  
    mask_inv_buf = mask_inv_buf & ~mask_buf; // This shows "holes" as white regions

    // Cycle over nlinfo and check for holes in columns with more than 1 nodepair
    std::vector<Array2D<double>> sub_boundaries; 
    for (difference_type nl_idx = 0; nl_idx < nodelist.width(); ++nl_idx) {
        if (noderange(nl_idx) > 2) {
            difference_type p2 = nl_idx + left_nl;            
            // Test one pixel below bottom node of every node pair except 
            // for the last node pair, since this will be the bottom of the ROI
            for (difference_type np_idx = 0; np_idx < noderange(nl_idx) - 2; np_idx += 2) {
                difference_type np_bottom = nodelist(np_idx + 1,nl_idx);
                if (mask_inv_buf(np_bottom + 1,p2)) {
                    // This is a boundary which hasn't been analyzed yet
                    sub_boundaries.push_back(details::calc_boundary(mask_inv_buf, np_bottom + 0.5, p2 - 0.5));

                    // Fill in this boundary so it does not get analyzed again
                    // Note that sub_boundary cannot have a hole, so it is
                    // safe to use boundary fill. Also, boundary is guaranteed
                    // to fill region covered by nlinfo.
                    fill(mask_inv_buf, sub_boundaries.back(), false);  
                }
            }
        }
    }

    return ROI2D::region_boundary(std::move(add_boundary), std::move(sub_boundaries));  
}

// Utility -------------------------------------------------------------------//
void ROI2D::region_nlinfo::chk_nonempty_op(const std::string &op) const {
    if (this->empty()) {
        throw std::invalid_argument("Attempted to use " + op + " operator on empty nlinfo" + 
                                    ". nlinfo must be nonempty.");  
    }
}

// Static factory methods ----------------------------------------------------//
ROI2D::region_boundary ROI2D::region_boundary::load(std::ifstream &is) {
    // Form empty boundary then fill in values in accordance to how they are saved
    region_boundary boundary;
    
    // Load add boundary
    boundary.add = Array2D<double>::load(is);
    
    // Allocate new sub boundaries and then load them
    difference_type num_sub_boundaries = 0;
    is.read(reinterpret_cast<char*>(&num_sub_boundaries), std::streamsize(sizeof(difference_type)));
    boundary.sub.resize(num_sub_boundaries);
    for (auto &sub_boundary : boundary.sub) {
        sub_boundary = Array2D<double>::load(is);
    }
    
    return boundary;
}

// Operations interface ------------------------------------------------------//
std::ostream& operator<<(std::ostream &os, const ROI2D::region_boundary &boundary) {
    typedef ROI2D::difference_type                              difference_type;
        
    os << "Add Boundary: " << '\n' << boundary.add;
    for (difference_type boundary_idx = 0; boundary_idx < difference_type(boundary.sub.size()); ++boundary_idx) {
        os << '\n' << "Sub Boundary(" << boundary_idx << ") :" << '\n' << boundary.sub[boundary_idx];
    }
    
    return os;
}  

bool isequal(const ROI2D::region_boundary &boundary1, const ROI2D::region_boundary &boundary2) {
    typedef ROI2D::difference_type                              difference_type;
        
    if (isequal(boundary1.add,boundary2.add) && boundary1.sub.size() == boundary2.sub.size()) { 
        for (difference_type boundary_idx = 0; boundary_idx < difference_type(boundary1.sub.size()); ++boundary_idx) {
            if (!isequal(boundary1.sub[boundary_idx], boundary2.sub[boundary_idx])) {
                return false;
            }
        }
        // Add boundary and sub boundaries are the same
        return true;
    }
    
    return false;
}

void save(const ROI2D::region_boundary &boundary, std::ofstream &os) {    
    typedef ROI2D::difference_type                              difference_type;
    
    // Save add boundary -> vec of sub boundaries
    save(boundary.add, os);
    
    // Save length of vector
    difference_type num_sub_boundaries = boundary.sub.size();
    os.write(reinterpret_cast<const char*>(&num_sub_boundaries), std::streamsize(sizeof(difference_type)));
    // Save each sub boundary
    for (const auto &sub_boundary : boundary.sub) {
        save(sub_boundary, os);
    }
}

// Arithmetic methods --------------------------------------------------------//
ROI2D::region_nlinfo ROI2D::region_boundary::to_nlinfo(Array2D<bool> &mask_buf) const {        
    // Converts boundary to nlinfo. This will form an nlinfo corresponding to 
    // the largest contiguous enclosing region of the boundary.

    // Clear mask_buf and then draw boundary
    mask_buf() = false;
    fill(mask_buf, add, true);
    for (const auto &sub_boundary : sub) {
        fill(mask_buf, sub_boundary, false);
    }

    // Get nlinfos for this boundary
    auto nlinfos_pair = ROI2D::region_nlinfo::form_nlinfos(mask_buf);

    if (nlinfos_pair.first.empty()) {
        // Boundary is empty, so return empty nlinfo
        return ROI2D::region_nlinfo();
    }

    // Its possible for boundary to produce multiple nlinfos if it gets "pinched",
    // so return the nlinfo with the largest number of points. nlinfo is 
    // non-empty at this point, so max element is guaranteed to exist.
    return *std::max_element(nlinfos_pair.first.begin(), nlinfos_pair.first.end(), [](const ROI2D::region_nlinfo &a, const ROI2D::region_nlinfo &b) { return a.points < b.points; });
}

// Static factory methods ----------------------------------------------------//
ROI2D::region ROI2D::region::load(std::ifstream &is) {
    // Form empty region then fill in values in accordance to how they are saved
    region reg;
    
    // Load nlinfo
    reg.nlinfo = ROI2D::region_nlinfo::load(is);
    
    // Load boundary
    reg.boundary = ROI2D::region_boundary::load(is);
    
    return reg;
}

// Operations interface ------------------------------------------------------//
std::ostream& operator<<(std::ostream &os, const ROI2D::region &reg) {
    os << reg.nlinfo;             
    os << reg.boundary;             
    
    return os;
}  

void save(const ROI2D::region &reg, std::ofstream &os) {        
    // Save nlinfo -> boundary        
    save(reg.nlinfo, os);
    save(reg.boundary, os);
}

namespace details {     
    // Arithmetic methods ----------------------------------------------------//
    nlinfo_incrementor& nlinfo_incrementor::operator++() {        
        if (p1 == nlinfo_ptr->nodelist(np_idx+1,nl_idx)) {
            // incremented position will go beyond bottom node
            if (np_idx == nlinfo_ptr->noderange(nl_idx)-2) {
                // Incremented nodepair index will go beyond nodepairs in this column,
                // so search columns to find next one with nodepairs
                for (++nl_idx; nl_idx < nlinfo_ptr->nodelist.width() && nlinfo_ptr->noderange(nl_idx) == 0; ++nl_idx) { }
                if (nl_idx == nlinfo_ptr->nodelist.width()) {
                    // End has been reached, increment nodepair_idx and p1 to one
                    // beyond the end to match the end incrementor.
                    np_idx += 2;
                    ++p1;
                } else {
                    np_idx = 0;
                    p1 = nlinfo_ptr->nodelist(np_idx,nl_idx); // top node
                }
            } else {
                np_idx += 2;
                p1 = nlinfo_ptr->nodelist(np_idx,nl_idx); // top node
            }
        } else {
            ++p1;
        }
        
        return *this;
    }
        
    // Additional Constructors -----------------------------------------------//
    ROI2D_incrementor::ROI2D_incrementor(const ROI2D &roi, difference_type region_idx, const ROI2D::region_nlinfo::incrementor &nlinfo_inc) : roi(roi), region_idx(region_idx), nlinfo_inc(nlinfo_inc) {
        // If the initial nlinfo_inc is the beginning incrementor for the first
        // region, increment until the first position is found. This is done for 
        // convenience so the begin_inc() can be set easily by the caller.
        if (this->roi.size_regions() > 0 && this->nlinfo_inc == this->roi.get_nlinfo(0).begin_inc() && this->roi.get_nlinfo(0).empty()) {
            // Set to first non-empty region
            for (++this->region_idx; this->region_idx < this->roi.size_regions() && this->roi.get_nlinfo(this->region_idx).empty(); ++this->region_idx) { }
            if (this->region_idx == this->roi.size_regions()) {
                // All regions are empty - set nlinfo incrementor to the end
                // incrementor of the last region in order to match the end 
                // incrementor
                this->nlinfo_inc = this->roi.get_nlinfo(this->region_idx-1).end_inc();
            } else {
                this->nlinfo_inc = this->roi.get_nlinfo(region_idx).begin_inc();
            }
        }
    }
    
    // Arithmetic methods ----------------------------------------------------//
    ROI2D_incrementor& ROI2D_incrementor::operator++() {
        if (nlinfo_inc.pos_2D().first == roi.get_nlinfo(region_idx).last_pos_p1() && nlinfo_inc.pos_2D().second == roi.get_nlinfo(region_idx).last_pos_p2()) {
            // Reached last position in this region, increment region_idx until 
            // nonempty region is found - or until all regions have been checked
            for (++region_idx; region_idx < roi.size_regions() && roi.get_nlinfo(region_idx).empty(); ++region_idx) { }
            if (region_idx == roi.size_regions()) {
                // This is the end, set nlinfo_inc to end_incrementor of last 
                // region to match the end incrementor.
                nlinfo_inc = roi.get_nlinfo(region_idx-1).end_inc();
            } else {
                nlinfo_inc = roi.get_nlinfo(region_idx).begin_inc();
            }
        } else {
            ++nlinfo_inc;
        }
        
        return *this;
    }
    
    // Additional Constructors -----------------------------------------------//
    ROI2D_contig_subregion_generator::ROI2D_contig_subregion_generator(const ROI2D &roi, SUBREGION subregion_type, difference_type r) : roi(roi), r(r) {        
        // Find max height of nodelist in roi so nodelist can be set.
        difference_type max_height = 0;
        for (difference_type region_idx = 0; region_idx < this->roi.size_regions(); ++region_idx) {
            if (this->roi.get_nlinfo(region_idx).nodelist.height() > max_height) {
                max_height = this->roi.get_nlinfo(region_idx).nodelist.height();
            }
        }
        
        // Set buffers. These buffers are the upperbound for the sizes, so they 
        // never need to be resized when forming contiguous region. 
        this->nlinfo_subregion.nodelist = Array2D<difference_type>(max_height, 2*this->r + 1);
        this->nlinfo_subregion.noderange = Array2D<difference_type>(1, 2*this->r + 1);
        this->active_nodepairs = Array2D<bool>(max_height/2, 2*this->r + 1);
        
        // Note that simple_nlinfo must be 'simple' (look at the requirements
        // for a 'simple' ROI at the simple_* static factory method definitions).
        switch (subregion_type) {
            case SUBREGION::CIRCLE:
                this->nlinfo_simple = ROI2D::simple_circle(r).get_nlinfo(0); // Safe since ROI is simple
                break;
            case SUBREGION::SQUARE:
                this->nlinfo_simple = ROI2D::simple_square(r).get_nlinfo(0); // Safe since ROI is simple
                break;
        }
    }
    
    // Local function --------------------------------------------------------//
    void simple_contig_subregion_add_interacting_nodes(ROI2D::difference_type np_adj_p2, 
                                                       ROI2D::difference_type np_loaded_top,
                                                       ROI2D::difference_type np_loaded_bottom,
                                                       const ROI2D::ROI2D::region_nlinfo &nlinfo_roi,
                                                       const ROI2D::ROI2D::region_nlinfo &nlinfo_simple,
                                                       Array2D<bool> &active_nodepairs,
                                                       std::stack<ROI2D::difference_type> &queue_np_idx) {
        typedef ROI2D::difference_type                          difference_type;
        
        // Make sure idx's are in range of both nlinfo_roi and nlinfo_simple
        difference_type nl_adj_idx = np_adj_p2 - nlinfo_roi.left_nl;
        difference_type nl_simple_idx = np_adj_p2 - nlinfo_simple.left_nl;
        if (nlinfo_roi.noderange.in_bounds(nl_adj_idx) && nlinfo_simple.noderange.in_bounds(nl_simple_idx)) {
            // Get simple nodepair at adjacent position - safe since nlinfo is 'simple'
            difference_type np_simple_top = nlinfo_simple.nodelist(0, nl_simple_idx);     
            difference_type np_simple_bottom = nlinfo_simple.nodelist(1, nl_simple_idx);  
            // Scans nodes from top to bottom
            for (difference_type np_adj_idx = 0; np_adj_idx < nlinfo_roi.noderange(nl_adj_idx); np_adj_idx += 2) {
                difference_type np_adj_top = nlinfo_roi.nodelist(np_adj_idx, nl_adj_idx);
                difference_type np_adj_bottom = nlinfo_roi.nodelist(np_adj_idx + 1, nl_adj_idx);
                if (np_loaded_bottom < np_adj_top || np_loaded_bottom < np_simple_top) {
                    return; // top node of adjacent nodepair or simple nodepair is below bottom node of loaded nodepair
                } 
                if (active_nodepairs(np_adj_idx/2, nl_simple_idx) && np_loaded_top <= np_adj_bottom && np_loaded_top <= np_simple_bottom) {
                    // At this point, loaded nodepair interacts with both adjacent
                    // nodepair and simple nodepair. Take the union of the 
                    // intersection. Note that it is possible for adjacent nodepair
                    // and simple nodepair to be disjoint, which results in a 
                    // "flipped" nodepair, so test for it.
                    difference_type np_top = std::max(np_adj_top, np_simple_top);
                    difference_type np_bottom = std::min(np_adj_bottom, np_simple_bottom);
                    if (np_top <= np_bottom) {
                        // Inactivate node pair, and then insert into the queue.
                        // Since simple_nlinfo only contains two nodes per column, 
                        // a nodepair in nlinfo_roi can only interact with it
                        // once, so deactivating it is safe.
                        active_nodepairs(np_adj_idx/2, nl_simple_idx) = false;
                        queue_np_idx.push(np_top);
                        queue_np_idx.push(np_bottom);
                        queue_np_idx.push(np_adj_p2);
                    }
                }
            }
        }
    }            
        
    // Arithmetic methods ----------------------------------------------------//
    const ROI2D::region_nlinfo& ROI2D_contig_subregion_generator::operator()(difference_type p1, difference_type p2) const {        
        // Clear/initialize values in nlinfo_output - must do this here in case 
        // empty nlinfo_output is returned
        nlinfo_subregion.top = roi.height()-1;           // Gets updated
        nlinfo_subregion.bottom = 0;                     // Gets updated
        nlinfo_subregion.left = roi.width()-1;           // Gets updated
        nlinfo_subregion.right = 0;                      // Gets updated
        nlinfo_subregion.left_nl = p2 - r;               // Correct
        nlinfo_subregion.right_nl = p2 + r;              // Correct
        nlinfo_subregion.noderange() = 0;                // Gets updated
        nlinfo_subregion.points = 0;                     // Gets updated

        // Get region idx containing (p1,p2)
        auto region_idx_pair = roi.get_region_idx(p1, p2);
        if (region_idx_pair.first == -1) {
            // ROI does not contain the x,y coordinate - return the empty nlinfo_output
            return nlinfo_subregion;
        }
        
        // Set active_nodepairs to true
        active_nodepairs() = true;
                
        // Get nlinfo corresponding to p1 and p2
        auto &nlinfo_roi = roi.get_nlinfo(region_idx_pair.first);

        // Shift nlinfo_simple's position in-place
        nlinfo_simple.shift(p1 - r, p2 - r);

        // Get node pairs containing x and y, take their union with simple 
        // nodepair, and then add to queue.
        active_nodepairs(region_idx_pair.second/2, r) = false; // Inactivate nodepair    
        std::stack<difference_type> queue_np_idx; // Holds all nodepairs (along with their index) which need to be processed
        queue_np_idx.push(std::max(nlinfo_simple.nodelist(0, r), nlinfo_roi.nodelist(region_idx_pair.second, p2 - nlinfo_roi.left_nl)));     // Top
        queue_np_idx.push(std::min(nlinfo_simple.nodelist(1, r), nlinfo_roi.nodelist(region_idx_pair.second + 1, p2 - nlinfo_roi.left_nl))); // Bottom
        queue_np_idx.push(p2);                                                                                                               // idx
        while (!queue_np_idx.empty()) {
            // Pop nodepair and its position out of queue and compare 
            // it to adjacent nodepairs (left and right of np_loaded_p2)
            difference_type np_loaded_p2 = queue_np_idx.top(); queue_np_idx.pop();
            difference_type np_loaded_bottom = queue_np_idx.top(); queue_np_idx.pop();
            difference_type np_loaded_top = queue_np_idx.top(); queue_np_idx.pop();

            // Compare to node pairs LEFT. Any node pairs which interact are added to the queue
            details::simple_contig_subregion_add_interacting_nodes(np_loaded_p2 - 1, 
                                                                   np_loaded_top, 
                                                                   np_loaded_bottom, 
                                                                   nlinfo_roi, 
                                                                   nlinfo_simple,
                                                                   active_nodepairs,
                                                                   queue_np_idx);

            // Compare to node pairs RIGHT. Any node pairs which interact are added to the queue
            details::simple_contig_subregion_add_interacting_nodes(np_loaded_p2 + 1, 
                                                                   np_loaded_top, 
                                                                   np_loaded_bottom, 
                                                                   nlinfo_roi, 
                                                                   nlinfo_simple,
                                                                   active_nodepairs,
                                                                   queue_np_idx);

            // Update points
            nlinfo_subregion.points += np_loaded_bottom - np_loaded_top + 1;

            // Update bounds
            if (np_loaded_top < nlinfo_subregion.top) { nlinfo_subregion.top = np_loaded_top; }               // Top
            if (np_loaded_bottom > nlinfo_subregion.bottom) { nlinfo_subregion.bottom = np_loaded_bottom; }   // Bottom
            if (np_loaded_p2 < nlinfo_subregion.left) { nlinfo_subregion.left = np_loaded_p2; }               // Left
            if (np_loaded_p2 > nlinfo_subregion.right) { nlinfo_subregion.right = np_loaded_p2; }             // Right

            // Insert node pairs and then sort 
            difference_type nl_output_idx = np_loaded_p2 - p2 + r;
            nlinfo_subregion.nodelist(nlinfo_subregion.noderange(nl_output_idx), nl_output_idx) = np_loaded_top;
            nlinfo_subregion.nodelist(nlinfo_subregion.noderange(nl_output_idx) + 1, nl_output_idx) = np_loaded_bottom;
            std::sort(&nlinfo_subregion.nodelist(0, nl_output_idx), &nlinfo_subregion.nodelist(0, nl_output_idx) + nlinfo_subregion.noderange(nl_output_idx) + 2);
            
            // Update noderange
            nlinfo_subregion.noderange(nl_output_idx) += 2;
        }

        // Shift nlinfo_simple's position back to original place
        nlinfo_simple.shift(r - p1, r - p2);

        return nlinfo_subregion;
    }
}

}