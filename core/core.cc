#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include <vector>
#include <fstream>
#include <iterator>
#include <core.h>


//normalize segments (inplace) traces
void CORE::normalize_segment(int ti, int tj) {
	std::vector<float> tmp;

	for (int i = ti*kernel_size; i < ti*kernel_size+kernel_size; i++)
		for (int j = tj*kernel_size; j < ti*kernel_size+kernel_size; j++)
			tmp.push_back(traces[i][j]);

	float max = *(max_element(tmp.begin(), tmp.end()));
	float avg = accumulate(tmp.begin(), tmp.end(), 0.0)/tmp.size();
	
	for (int i = ti*kernel_size; i < ti*kernel_size+kernel_size; i++)
		for (int j = tj*kernel_size; j < ti*kernel_size+kernel_size; j++)
			traces[i][j] = (traces[i][j]-avg)/max;

}

void CORE::set_traces(std::vector<std::vector<float>> tr) {
	traces = tr;
}

void CORE::load_seg_w(std::string filename) {
	std::ifstream file;
	float a;
	int c = 0;
	std::string line;
	file.open(filename);
	while (std::getline(file, line)) {
		std::istringstream in(line);
		traces.push_back(std::vector<float>());
		while(in >> a) {
			in >> a;
			traces[c].push_back(a);
		}
		c++;
	}
	file.close();
}

std::vector<std::vector<float>> CORE::predict_crack() {
	int w = traces[0].size();
	int h = traces.size();

	int it_w  = ((w/kernel_size)-1) < 0 ? 0 : ((w/kernel_size)-1);
	int it_h = (h/kernel_size)-1 < 0 ? 0 : ((h/kernel_size)-1);

	for (int i = 0; i < it_w; i++) {
		for (int j = 0; j < it_h; j++) {
			normalize_segment(i, j);
		}
	}

	return traces;
}

void CORE::write_to_file(std::string filename) {
	std::ofstream file;
	file.open(filename);
	
	std::vector<std::vector<float>>::iterator row;
	std::vector<float>::iterator col;

	for (row = traces.begin(); row != traces.end(); row++) {
		for (col = row->begin(); col != row->end(); col++) {
			file << *col << " ";
		}
		file << "\n";
	}
}

