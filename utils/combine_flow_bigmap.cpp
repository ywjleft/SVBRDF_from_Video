#include <cmath>
#include <algorithm>
#include <unistd.h>
#include <iostream>

void interp(double* input_flows, double y, double x, double* interped) {
	int res = 1024;
	int x0 = int(floor(x));
	int x1 = x0 + 1;
	int y0 = int(floor(y));
	int y1 = y0 + 1;
	double ax = x - x0;
	double ay = y - y0;
	double f00 = input_flows[(y0 * res + x0) * 2];
	double f01 = input_flows[(y0 * res + x0 + 1) * 2];
	double f10 = input_flows[(y0 * res + x0 + res) * 2];
	double f11 = input_flows[(y0 * res + x0 + res + 1) * 2];
	interped[0] = f00 * (1 - ay) * (1 - ax) + f01 * (1 - ay) * ax + f10 * ay * (1 - ax) + f11 * ay * ax;
	f00 = input_flows[(y0 * res + x0) * 2 + 1];
	f01 = input_flows[(y0 * res + x0 + 1) * 2 + 1];
	f10 = input_flows[(y0 * res + x0 + res) * 2 + 1];
	f11 = input_flows[(y0 * res + x0 + res + 1) * 2 + 1];
	interped[1] = f00 * (1 - ay) * (1 - ax) + f01 * (1 - ay) * ax + f10 * ay * (1 - ax) + f11 * ay * ax;
}

extern "C" void combine_flow(double* input_flow_bigmap, int h, int w, int y0, int x0, double* input_flows, int count, double* combined_flow) {
	int res = 1024;
	for (int i = 0; i < h * w * 2; i++) {
		combined_flow[i] = input_flow_bigmap[i];
	}
	for (int i = 0; i < count; i++) {
		//int counter1 = 0;
		//int counter2 = 0;

		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				if (std::abs(combined_flow[(y * w + x) * 2]) > 1e10) {
					continue;
				}
				double interped[2];
				double newy = y - y0 - combined_flow[(y * w + x) * 2 + 1];
				double newx = x - x0 - combined_flow[(y * w + x) * 2];
				//counter1++;
				//std::cout << "yx: " << y << ',' << x << ',' << combined_flow[(y * res + x) * 2 + 1] << ',' << combined_flow[(y * res + x) * 2] << std::endl;
				if (newy > res - 1 || newy < 0 || newx > res - 1 || newx < 0) {
					combined_flow[(y * w + x) * 2] = 1e20;
					combined_flow[(y * w + x) * 2 + 1] = 1e20;
					continue;
				}
				//counter2++;
				interp(input_flows + i * res * res * 2, newy, newx, interped);
				//std::cout << newy << ',' << newx << ',' << interped[0] << ',' << interped[1] << std::endl;
				//usleep(100);
				combined_flow[(y * w + x) * 2] = combined_flow[(y * w + x) * 2] + interped[0];
				combined_flow[(y * w + x) * 2 + 1] = combined_flow[(y * w + x) * 2 + 1] + interped[1];
			}
		}
		//std::cout << i << ':' << counter1 << ',' << counter2 << std::endl;
	}
}