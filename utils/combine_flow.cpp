#include <cmath>
#include <algorithm>

void interp(double* input_flows, double y, double x, double* interped) {
	int res = 256;
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

extern "C" void combine_flow(double* input_flows, int count, int* mask, double* combined_flow) {
	int res = 256;
	for (int i = 0; i < res * res * 2; i++) {
		combined_flow[i] = 0.0f;
	}
	for (int i = 0; i < res * res; i++) {
		mask[i] = 1;
	}
	for (int i = 0; i < count; i++) {
		for (int y = 0; y < res; y++) {
			for (int x = 0; x < res; x++) {
				if (mask[y * res + x] == 0) {
					continue;
				}
				double interped[2];
				double newy = y - combined_flow[(y * res + x) * 2 + 1];
				double newx = x - combined_flow[(y * res + x) * 2];
				if (newy > res - 1 || newy < 0 || newx > res - 1 || newx < 0) {
					mask[y * res + x] = 0;
					combined_flow[(y * res + x) * 2] = 1e20;
					combined_flow[(y * res + x) * 2 + 1] = 1e20;
					continue;
				}
				interp(input_flows + i * res * res * 2, newy, newx, interped);
				combined_flow[(y * res + x) * 2] = combined_flow[(y * res + x) * 2] + interped[0];
				combined_flow[(y * res + x) * 2 + 1] = combined_flow[(y * res + x) * 2 + 1] + interped[1];
			}
		}
	}
}