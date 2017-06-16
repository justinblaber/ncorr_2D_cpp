#include "ncorr.h"

using namespace ncorr;

int main(int argc, char *argv[]) {
	if (argc != 2) {
		throw std::invalid_argument("Must have 1 command line input of either 'calculate' or 'load'");	
	}

	// Initialize DIC and strain information ---------------//
	DIC_analysis_input DIC_input;
	DIC_analysis_output DIC_output;
	strain_analysis_input strain_input;
	strain_analysis_output strain_output;

	// Determine whether or not to perform calculations or 
	// load data (only load data if analysis has already 
	// been done and saved or else throw an exception).
	std::string input(argv[1]);
	if (input == "load") {
		// Load inputs
		DIC_input = DIC_analysis_input::load("save/DIC_input.bin");
		DIC_output = DIC_analysis_output::load("save/DIC_output.bin");
		strain_input = strain_analysis_input::load("save/strain_input.bin");
		strain_output = strain_analysis_output::load("save/strain_output.bin");
	} else if (input == "calculate") {
		// Set images
		std::vector<Image2D> imgs;
		for (int i = 0; i <= 11; ++i) {
		    std::ostringstream ostr;
		    ostr << "images/ohtcfrp_" << std::setfill('0') << std::setw(2) << i << ".png";
		    imgs.push_back(ostr.str());
		}
		
		// Set DIC_input
		DIC_input = DIC_analysis_input(imgs, 							// Images
				               ROI2D(Image2D("images/roi.png").get_gs() > 0.5),		// ROI
					       3,                                         		// scalefactor
					       INTERP::QUINTIC_BSPLINE_PRECOMPUTE,			// Interpolation
					       SUBREGION::CIRCLE,					// Subregion shape
					       20,                                        		// Subregion radius
					       4,                                         		// # of threads
					       DIC_analysis_config::NO_UPDATE,				// DIC configuration for reference image updates
					       true);							// Debugging enabled/disabled

		// Perform DIC_analysis    
		DIC_output = DIC_analysis(DIC_input);

		// Convert DIC_output to Eulerian perspective
		DIC_output = change_perspective(DIC_output, INTERP::QUINTIC_BSPLINE_PRECOMPUTE);

		// Set units of DIC_output (provide units/pixel)
		DIC_output = set_units(DIC_output, "mm", 0.2);

		// Set strain input
		strain_input = strain_analysis_input(DIC_input,
		                                     DIC_output,
		                                     SUBREGION::CIRCLE,					// Strain subregion shape
		                                     5);						// Strain subregion radius
		
		// Perform strain_analysis
		strain_output = strain_analysis(strain_input); 
		
		// Save outputs as binary
                save(DIC_input, "save/DIC_input.bin");
                save(DIC_output, "save/DIC_output.bin");
                save(strain_input, "save/strain_input.bin");
                save(strain_output, "save/strain_output.bin");
	} else {
		throw std::invalid_argument("Input of " + input + " is not recognized. Must be either 'calculate' or 'load'");	
	}		
        
        // Create Videos ---------------------------------------//
	// Note that more inputs can be used to modify plots. 
	// If video is not saving correctly, try changing the 
	// input codec using cv::VideoWriter::fourcc(...)). Check 
	// the opencv documentation on video codecs. By default, 
	// ncorr uses cv::VideoWriter::fourcc('M','J','P','G')).
        save_DIC_video("video/test_v_eulerian.avi", 
                       DIC_input, 
                       DIC_output, 
                       DISP::V,
                       0.5,		// Alpha		
                       15);		// FPS

        save_DIC_video("video/test_u_eulerian.avi", 
                       DIC_input, 
                       DIC_output, 
                       DISP::U, 
                       0.5,		// Alpha
                       15);		// FPS

        save_strain_video("video/test_eyy_eulerian.avi", 
                          strain_input, 
                          strain_output, 
                          STRAIN::EYY, 
                          0.5,		// Alpha
                          15);		// FPS

        save_strain_video("video/test_exy_eulerian.avi", 
                          strain_input, 
                          strain_output, 
                          STRAIN::EXY, 
                          0.5,		// Alpha
                          15);		// FPS
        
        save_strain_video("video/test_exx_eulerian.avi", 
                          strain_input, 
                          strain_output, 
                          STRAIN::EXX, 
                          0.5,		// Alpha
                          15); 		// FPS

  	return 0;
}
