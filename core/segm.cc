#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include <vector>
#include <fstream>
#include <iterator>

using namespace tensorflow;

const int size = 64;

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

void write_file_out(std::string filename, std::vector<std::vector<float>> &out) {
	std::ofstream file;
	file.open(filename);

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			file << out[i][j] << " ";
		}
		file << "\n";
	}

}

void load_seg_w64(std::string filename, std::vector<std::vector<float>> &w) {
	std::ifstream file;
	float a;
	int c = 0;
	std::string line;
	file.open(filename);
	while (std::getline(file, line)) {
		std::istringstream in(line);
		for (int i = 0; i < size; i++) {
			in >> a;
			w[c][i] = a;
		}
		c++;
	}
	file.close();
}

void normalize(std::vector<std::vector<float>> &input) {
	std::vector<float> tmp;

	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			tmp.push_back(input[i][j]);

	float max = *(max_element(tmp.begin(), tmp.end()));
	float avg = accumulate(tmp.begin(), tmp.end(), 0.0)/tmp.size();
	
	std::cout << tmp[0] << max << " " << avg << std::endl;

	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			input[i][j] = (input[i][j]-avg)/max;
}


int main(int argc, char* argv[]) {
	Session* session;
	Status status = NewSession(SessionOptions(), &session);
	if (!status.ok()) {
		std::cout << status.ToString() << "\n";
		return 1;
	}

	GraphDef graph_def;
	status = ReadBinaryProto(Env::Default(), argv[1], &graph_def);
	if (!status.ok()) {
		std::cout << status.ToString() << "\n";
		return 1;
	}

	status = session->Create(graph_def);
	if (!status.ok()) {
		std::cout << status.ToString() << "\n";
		return 1;
	}

	Tensor inputTensor (DT_FLOAT, TensorShape({64, 64}));

	std::vector<std::vector<float>> ten(64, std::vector<float>(64, 0.0));
        load_seg_w64("tsegy", ten);
	normalize(ten);
	write_file_out("norm_tseg", ten); // test normalization

	for (int i = 0; i < 64; i++)
		for (int j = 0; j < 64; j++)
			inputTensor.matrix<float>()(i, j) = ten[i][j];

	std::vector<tensorflow::Tensor> output;

	status = session->Run({{"input_1", inputTensor} }, { "decoder/conv2d_6/truediv" }, {}, &output);
	if (!status.ok()) {
		std::cout << status.ToString() << "\n";
		return 1;
	}

	auto output_c = output[0].scalar<float>();

	std::cout << output_c() << "\n";

	session->Close();
	return 0;
}

