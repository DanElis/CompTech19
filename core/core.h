#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include <vector>
#include <fstream>
#include <iterator>

using namespace tensorflow;

class CORE {
	public:
		void normalize_segment(int ti, int tj);
		void load_seg_w(std::string filename);
		void write_to_file(std::string filename);
		void set_traces(std::vector<std::vector<float>> traces);
		std::vector<std::vector<float>> predict_crack();
		std::vector<std::vector<float>> get_traces();

	private:
		int kernel_size = 64;
		std::vector<std::vector<float>> traces;
};

