#include <iostream>

#include "netikat.h"

int main(int argc, char **argv) {
  PacketType pType{5 /* node */, 2 /* dest */};
  NetiKAT<double> neti(pType, 2);

  TransitionMatrix<double> node0Prog =
      neti.choice(0.5, neti.set(0, 1), neti.set(0, 4));
  TransitionMatrix<double> node1Prog = neti.skip();
  TransitionMatrix<double> node2Prog =
      neti.choice(0.5, neti.set(0, 3), neti.set(0, 4));
  TransitionMatrix<double> node3Prog = neti.skip();
  TransitionMatrix<double> node4Prog =
      neti.amp(neti.ifThen(neti.test(1, 0), neti.set(0, 1)),
               neti.ifThen(neti.test(1, 1), neti.set(0, 3)));
  TransitionMatrix<double> done = neti.amp(neti.test(0, 1), neti.test(0, 3));
  TransitionMatrix<double> notDone =
      neti.amp(std::vector<TransitionMatrix<double>>{
          neti.test(0, 0), neti.test(0, 2), neti.test(0, 4)});
  // neti.seq(neti.amp(neti.ifThen(neti.test(1, 0), neti.set(0, 1)),
  //                   neti.ifThen(neti.test(1, 1), neti.set(0, 3))),
  //          neti.cap1());

  TransitionMatrix<double> step =
      neti.amp(std::vector<TransitionMatrix<double>>{
          neti.ifThen(neti.test(0, 0), node0Prog),
          neti.ifThen(neti.test(0, 1), node1Prog),
          neti.ifThen(neti.test(0, 2), node2Prog),
          neti.ifThen(neti.test(0, 3), node3Prog),
          neti.ifThen(neti.test(0, 4), node4Prog)});

  TransitionMatrix<double> prog = step;
  // TransitionMatrix<double> prog = neti.seq(step, step);
  // TransitionMatrix<double> prog = neti.seq(done, neti.seq(step, step));
  // neti.seq(done, neti.starApprox(neti.seq(step, notDone)));

  auto pkt = [&](size_t a, size_t b) {
    return neti.packetIndex(std::vector<size_t>{a, b});
  };

  Distribution<double> v = neti.toVec(PacketSet{pkt(0, 0), pkt(2, 1)});
  Distribution<double> vOut = prog(v);

  auto printPacket = [&](size_t packetIndex) {
    Packet p = neti.packetFromIndex(packetIndex);
    cout << "(";
    for (size_t iF = 0; iF < p.size(); ++iF) {
      cout << p[iF];
      if (iF < p.size() - 1) {
        cout << ", ";
      }
    }
    cout << ")";
  };
  auto printPacketSet = [&](size_t setIndex) {
    PacketSet ps = neti.packetSetFromIndex(setIndex);
    cout << "{";
    size_t iPS = 0;
    for (size_t pIdx : ps) {
      printPacket(pIdx);
      if (iPS++ < ps.size() - 1) {
        cout << ", ";
      }
    }
    cout << "}";
  };
  auto printDistribution = [&](Distribution<double> d) {
    for (std::pair<size_t, double> iP : d) {
      printPacketSet(iP.first);
      cout << "\t" << iP.second << endl;
    }
  };

  cout << "firstStep: " << endl;
  printDistribution(step(v));

  cout << "test4(step(v)): " << endl;
  printDistribution(neti.test(0, 4)(step(v)));

  cout << "node4prog(step(v)): " << endl;
  Distribution<double> prog4 = neti.ifThen(neti.test(0, 4), node4Prog)(step(v));
  printDistribution(prog4);

  cout << "node1prog(step(v)): " << endl;
  Distribution<double> prog1 = neti.ifThen(neti.test(0, 1), node1Prog)(step(v));
  printDistribution(prog1);

  cout << "prog4 & prog1" << endl;
  printDistribution(neti.amp(prog1, prog4));

  cout << "step step" << endl;
  printDistribution(step(step(v)));

  // Eigen::VectorXd outputDist = toVec(vOut);
  // std::cout << "Output Distribution: " << endl << outputDist << endl;
  return 0;
}
