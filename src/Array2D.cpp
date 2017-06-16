/* 
 * File:   Array2D.h
 * Author: justin
 *
 * Created on December 30, 2014, 4:55 PM
 */

#include "Array2D.h"

namespace ncorr {
    
namespace details {     
    // The only thread-safe routine in FFTW is fftw_execute(), so use this mutex
    // for all other routines.
    std::mutex fftw_mutex;    
}
    
}