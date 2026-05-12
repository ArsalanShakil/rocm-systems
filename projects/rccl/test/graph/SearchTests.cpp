/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Regression test for the rail-matching helper ncclTopoSearchCheckNet() in
// src/graph/search.cc, which decides whether a candidate NIC is a valid
// "back-to-NIC" choice during ring/tree topology search.
//
// Targets the NCCL v2.28.7-1 fix for NCCL_CROSS_NIC=0 ("cannot find a viable
// ring"). The old code matched rails by (asic, port) only, which fails across
// hosts when NICs have unique per-host GUIDs. The fix adds a per-NIC `pciId`
// and NCCL_MNNVL_RAIL_PER_HOST: when set, crossNic=0 matches cross-host NICs by
// (pciId, port) instead of (asic, port). (Synced into RCCL by the v2.28.9-1
// sync; requires net.pciId in topo.h and a non-static helper in search.cc, so
// pre-sync trees cannot even compile this file.)
//
// We pin the helper against a fabricated ncclTopoSystem/ncclTopoGraph across
// pattern (TREE/RING/BALANCED_TREE) x crossNic (0/1/2) x same/different system
// id x matching/mismatching asic/port/pciId x NCCL_MNNVL_RAIL_PER_HOST (0/1).
//
// Each body runs via RUN_ISOLATED_TEST(_WITH_ENV): NCCL_PARAM caches its value
// in a function-local static that is COW-inherited across fork(), so every case
// forks to keep the parent's ncclParamMnnvlRailPerHost() cache untouched.

#include "graph.h"
#include "graph/topo.h"
#include "gtest/gtest.h"

#include "../common/ProcessIsolatedTestRunner.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

// Forward declaration of the function under test. See prerequisite (c) above.
bool ncclTopoSearchCheckNet(struct ncclTopoSystem* system,
                            struct ncclTopoGraph*  graph,
                            struct ncclTopoNode*   startNet,
                            int                    n,
                            int                    step);

namespace RcclUnitTesting
{
namespace
{

struct NetSpec
{
    int      systemId;
    int      dev;
    uint64_t asic;
    int      port;
    uint64_t pciId;
};

// Allocates a zero-initialised ncclTopoSystem on the heap and populates the
// NET node array from `nics`. The system has no GPUs / paths / links — the
// helper under test only reads `system->nodes[NET].nodes[i]`, so that's all we
// need to set up.
ncclTopoSystem* makeSystemWithNets(const std::vector<NetSpec>& nics)
{
    auto* sys             = new ncclTopoSystem{};
    sys->nodes[NET].count = static_cast<int>(nics.size());
    for(size_t i = 0; i < nics.size(); ++i)
    {
        auto& node     = sys->nodes[NET].nodes[i];
        node.type      = NET;
        node.id        = NCCL_TOPO_ID(nics[i].systemId, nics[i].dev);
        node.net.dev   = nics[i].dev;
        node.net.asic  = nics[i].asic;
        node.net.port  = nics[i].port;
        node.net.pciId = nics[i].pciId; // requires prerequisite (a)
    }
    return sys;
}

ncclTopoGraph* makeGraph(int pattern, int crossNic, int nChannels = 0)
{
    auto* g      = new ncclTopoGraph{};
    g->pattern   = pattern;
    g->crossNic  = crossNic;
    g->nChannels = nChannels;
    return g;
}

} // namespace

// ----------------------------------------------------------------------------
// PATTERN_TREE: trees are symmetric, so the only valid back-NIC is startNet.
// ----------------------------------------------------------------------------
TEST(SearchCheckNet, Tree_SameId_Accepts)
{
    RUN_ISOLATED_TEST("Tree_SameId_Accepts",
                      []()
                      {
                          auto* sys      = makeSystemWithNets({
                              {0, 0, 0xAA, 1, 0xCC}
                          });
                          auto* g        = makeGraph(NCCL_TOPO_PATTERN_TREE, /*crossNic=*/1);
                          auto* startNet = &sys->nodes[NET].nodes[0];
                          EXPECT_TRUE(
                              ncclTopoSearchCheckNet(sys, g, startNet, /*n=*/0, /*step=*/0));
                          delete g;
                          delete sys;
                      });
}

TEST(SearchCheckNet, Tree_DifferentId_Rejects)
{
    RUN_ISOLATED_TEST("Tree_DifferentId_Rejects",
                      []()
                      {
                          auto* sys      = makeSystemWithNets({
                              {0, 0, 0xAA, 1, 0xCC},
                              {0, 1, 0xAA, 1, 0xCC},
                          });
                          auto* g        = makeGraph(NCCL_TOPO_PATTERN_TREE, /*crossNic=*/1);
                          auto* startNet = &sys->nodes[NET].nodes[0];
                          EXPECT_FALSE(
                              ncclTopoSearchCheckNet(sys, g, startNet, /*n=*/1, /*step=*/0));
                          delete g;
                          delete sys;
                      });
}

// ----------------------------------------------------------------------------
// PATTERN_RING with crossNic == 1 (the default): no rail constraint.
// ----------------------------------------------------------------------------
TEST(SearchCheckNet, Ring_CrossNic1_AcceptsAnyNet)
{
    RUN_ISOLATED_TEST("Ring_CrossNic1_AcceptsAnyNet",
                      []()
                      {
                          auto* sys      = makeSystemWithNets({
                              {0, 0, 0xAA, 1, 0xCC},
                              {1, 1, 0xBB, 2, 0xDD}, // wildly different across the board
                          });
                          auto* g        = makeGraph(NCCL_TOPO_PATTERN_RING, /*crossNic=*/1);
                          auto* startNet = &sys->nodes[NET].nodes[0];
                          EXPECT_TRUE(
                              ncclTopoSearchCheckNet(sys, g, startNet, /*n=*/1, /*step=*/0));
                          delete g;
                          delete sys;
                      });
}

// ----------------------------------------------------------------------------
// PATTERN_RING with crossNic == 2: alternating channels. Even channels accept
// anything; odd channels must come back to the previous channel's start NIC.
// ----------------------------------------------------------------------------
TEST(SearchCheckNet, Ring_CrossNic2_EvenChannel_Accepts)
{
    RUN_ISOLATED_TEST(
        "Ring_CrossNic2_EvenChannel_Accepts",
        []()
        {
            auto* sys      = makeSystemWithNets({
                {0, 0, 0xAA, 1, 0xCC},
                {0, 1, 0xBB, 2, 0xDD},
            });
            auto* g        = makeGraph(NCCL_TOPO_PATTERN_RING, /*crossNic=*/2, /*nChannels=*/0);
            auto* startNet = &sys->nodes[NET].nodes[0];
            EXPECT_TRUE(ncclTopoSearchCheckNet(sys, g, startNet, /*n=*/1, /*step=*/0));
            delete g;
            delete sys;
        });
}

TEST(SearchCheckNet, Ring_CrossNic2_OddChannel_MatchingPrevStart_Accepts)
{
    RUN_ISOLATED_TEST(
        "Ring_CrossNic2_OddChannel_MatchingPrevStart_Accepts",
        []()
        {
            auto* sys = makeSystemWithNets({
                {0, 0, 0xAA, 1, 0xCC},
                {0, 1, 0xBB, 2, 0xDD},
            });
            auto* g   = makeGraph(NCCL_TOPO_PATTERN_RING, /*crossNic=*/2, /*nChannels=*/1);
            g->inter[0]
                = sys->nodes[NET].nodes[1].id; // graph->inter[(nChannels-1)*2] for nChannels=1
            auto* startNet = &sys->nodes[NET].nodes[0];
            EXPECT_TRUE(ncclTopoSearchCheckNet(sys, g, startNet, /*n=*/1, /*step=*/0));
            delete g;
            delete sys;
        });
}

TEST(SearchCheckNet, Ring_CrossNic2_OddChannel_MismatchPrevStart_Rejects)
{
    RUN_ISOLATED_TEST(
        "Ring_CrossNic2_OddChannel_MismatchPrevStart_Rejects",
        []()
        {
            auto* sys      = makeSystemWithNets({
                {0, 0, 0xAA, 1, 0xCC},
                {0, 1, 0xBB, 2, 0xDD},
            });
            auto* g        = makeGraph(NCCL_TOPO_PATTERN_RING, /*crossNic=*/2, /*nChannels=*/1);
            g->inter[0]    = sys->nodes[NET].nodes[0].id; // require start net at index 0
            auto* startNet = &sys->nodes[NET].nodes[0];
            EXPECT_FALSE(ncclTopoSearchCheckNet(sys, g, startNet, /*n=*/1, /*step=*/0));
            delete g;
            delete sys;
        });
}

// ----------------------------------------------------------------------------
// PATTERN_RING with crossNic == 0, single host (all NICs on systemId=0).
// Legacy (asic, port) match should keep working untouched.
// ----------------------------------------------------------------------------
TEST(SearchCheckNet, Ring_CrossNic0_SameHost_AsicAndPortMatch_Accepts)
{
    RUN_ISOLATED_TEST(
        "Ring_CrossNic0_SameHost_AsicAndPortMatch_Accepts",
        []()
        {
            auto* sys      = makeSystemWithNets({
                {0, 0, 0xAA, 1, 0xCC},
                {0, 1, 0xAA, 1, 0xDD}, // same asic + port (twin-port NIC), distinct dev
            });
            auto* g        = makeGraph(NCCL_TOPO_PATTERN_RING, /*crossNic=*/0);
            auto* startNet = &sys->nodes[NET].nodes[0];
            EXPECT_TRUE(ncclTopoSearchCheckNet(sys, g, startNet, /*n=*/1, /*step=*/0));
            delete g;
            delete sys;
        });
}

TEST(SearchCheckNet, Ring_CrossNic0_SameHost_DifferentAsic_Rejects)
{
    RUN_ISOLATED_TEST("Ring_CrossNic0_SameHost_DifferentAsic_Rejects",
                      []()
                      {
                          auto* sys      = makeSystemWithNets({
                              {0, 0, 0xAA, 1, 0xCC},
                              {0, 1, 0xBB, 1, 0xCC},
                          });
                          auto* g        = makeGraph(NCCL_TOPO_PATTERN_RING, /*crossNic=*/0);
                          auto* startNet = &sys->nodes[NET].nodes[0];
                          EXPECT_FALSE(
                              ncclTopoSearchCheckNet(sys, g, startNet, /*n=*/1, /*step=*/0));
                          delete g;
                          delete sys;
                      });
}

TEST(SearchCheckNet, Ring_CrossNic0_SameHost_DifferentPort_Rejects)
{
    RUN_ISOLATED_TEST("Ring_CrossNic0_SameHost_DifferentPort_Rejects",
                      []()
                      {
                          auto* sys      = makeSystemWithNets({
                              {0, 0, 0xAA, 1, 0xCC},
                              {0, 1, 0xAA, 2, 0xCC},
                          });
                          auto* g        = makeGraph(NCCL_TOPO_PATTERN_RING, /*crossNic=*/0);
                          auto* startNet = &sys->nodes[NET].nodes[0];
                          EXPECT_FALSE(
                              ncclTopoSearchCheckNet(sys, g, startNet, /*n=*/1, /*step=*/0));
                          delete g;
                          delete sys;
                      });
}

// ----------------------------------------------------------------------------
// PATTERN_RING with crossNic == 0, cross-host, MNNVL_RAIL_PER_HOST default off.
// Legacy (asic, port) match is the only path. With unique-per-NIC GUIDs across
// hosts (the realistic case) the search is forced to reject — this is the
// "cannot find a viable ring" condition the upstream notes describe.
// ----------------------------------------------------------------------------
TEST(SearchCheckNet, Ring_CrossNic0_CrossHost_MnnvlRailOff_LegacyAsicMatch_Accepts)
{
    RUN_ISOLATED_TEST(
        "Ring_CrossNic0_CrossHost_MnnvlRailOff_LegacyAsicMatch_Accepts",
        []()
        {
            // Synthetic case: same asic/port across hosts (e.g. virtual/SR-IOV NICs
            // exposing identical GUIDs). With the legacy code this is the *only* way
            // a cross-host ring works under crossNic=0.
            auto* sys      = makeSystemWithNets({
                {/*systemId*/ 0, /*dev*/ 0, /*asic*/ 0xAA, /*port*/ 1, /*pciId*/ 0xCC},
                {/*systemId*/ 1, /*dev*/ 0, /*asic*/ 0xAA, /*port*/ 1, /*pciId*/ 0xDD},
            });
            auto* g        = makeGraph(NCCL_TOPO_PATTERN_RING, /*crossNic=*/0);
            auto* startNet = &sys->nodes[NET].nodes[0];
            EXPECT_TRUE(ncclTopoSearchCheckNet(sys, g, startNet, /*n=*/1, /*step=*/0));
            delete g;
            delete sys;
        });
}

TEST(SearchCheckNet, Ring_CrossNic0_CrossHost_MnnvlRailOff_UniqueGuids_Rejects)
{
    RUN_ISOLATED_TEST(
        "Ring_CrossNic0_CrossHost_MnnvlRailOff_UniqueGuids_Rejects",
        []()
        {
            // Realistic case: each NIC has its own globally-unique GUID, so asic
            // never matches across hosts. With MNNVL_RAIL_PER_HOST=0 the helper has
            // no other criterion to fall back on -> reject. This documents the
            // legacy (broken) behaviour that motivated the fix.
            auto* sys      = makeSystemWithNets({
                {/*systemId*/ 0, /*dev*/ 0, /*asic*/ 0xAA01, /*port*/ 1, /*pciId*/ 0xCAFE},
                {/*systemId*/ 1, /*dev*/ 0, /*asic*/ 0xAA02, /*port*/ 1, /*pciId*/ 0xCAFE},
            });
            auto* g        = makeGraph(NCCL_TOPO_PATTERN_RING, /*crossNic=*/0);
            auto* startNet = &sys->nodes[NET].nodes[0];
            EXPECT_FALSE(ncclTopoSearchCheckNet(sys, g, startNet, /*n=*/1, /*step=*/0));
            delete g;
            delete sys;
        });
}

// ----------------------------------------------------------------------------
// PATTERN_RING with crossNic == 0, cross-host, MNNVL_RAIL_PER_HOST=1 (the fix).
// Cross-host matching falls back to (pciId, port). NCCL_PARAM caches the env
// value via pthread_once, so each MNNVL test runs in its own forked process.
// ----------------------------------------------------------------------------
TEST(SearchCheckNet, Ring_CrossNic0_CrossHost_MnnvlRailOn_PciIdMatch_Accepts)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "MnnvlRailOn_PciIdMatch_Accepts",
        []()
        {
            auto* sys      = makeSystemWithNets({
                {        0,        0,         0xAA01,         1, 0xCAFE}, // unique GUIDs per host (realistic)
                {1, 0, 0xAA02, 1, 0xCAFE}, // identical pciId + port across hosts
            }
                    );
            auto* g        = makeGraph(NCCL_TOPO_PATTERN_RING, /*crossNic=*/0);
            auto* startNet = &sys->nodes[NET].nodes[0];
            EXPECT_TRUE(ncclTopoSearchCheckNet(sys, g, startNet, 1, 0));
            delete g;
            delete sys;
    },
        {{"NCCL_MNNVL_RAIL_PER_HOST", "1"}});
}

TEST(SearchCheckNet, Ring_CrossNic0_CrossHost_MnnvlRailOn_PciIdMismatch_Rejects)
{
    RUN_ISOLATED_TEST_WITH_ENV("MnnvlRailOn_PciIdMismatch_Rejects",
                               []()
                               {
                                   auto* sys = makeSystemWithNets({
                                       {        0,        0,         0xAA01,         1, 0xCAFE},
                                       {1, 0, 0xAA02, 1, 0xBABE}, // pciId differs => different rail
                                   }
                                      );
                                   auto* g   = makeGraph(NCCL_TOPO_PATTERN_RING, /*crossNic=*/0);
                                   auto* startNet = &sys->nodes[NET].nodes[0];
                                   EXPECT_FALSE(ncclTopoSearchCheckNet(sys, g, startNet, 1, 0));
                                   delete g;
                                   delete sys;
    },
                               {{"NCCL_MNNVL_RAIL_PER_HOST", "1"}});
}

TEST(SearchCheckNet, Ring_CrossNic0_CrossHost_MnnvlRailOn_PortMismatch_Rejects)
{
    RUN_ISOLATED_TEST_WITH_ENV("MnnvlRailOn_PortMismatch_Rejects",
                               []()
                               {
                                   auto* sys = makeSystemWithNets({
                                       {        0,        0,         0xAA01,         1, 0xCAFE},
                                       {1, 0, 0xAA02, 2, 0xCAFE}, // matching pciId, different port
                                   }
                                      );
                                   auto* g   = makeGraph(NCCL_TOPO_PATTERN_RING, /*crossNic=*/0);
                                   auto* startNet = &sys->nodes[NET].nodes[0];
                                   EXPECT_FALSE(ncclTopoSearchCheckNet(sys, g, startNet, 1, 0));
                                   delete g;
                                   delete sys;
    },
                               {{"NCCL_MNNVL_RAIL_PER_HOST", "1"}});
}

TEST(SearchCheckNet, Ring_CrossNic0_SameHost_MnnvlRailOn_StillUsesAsicMatch)
{
    // MNNVL_RAIL_PER_HOST only switches the matching key when the NICs are on
    // *different* system IDs. Same-host pairs still use (asic, port).
    RUN_ISOLATED_TEST_WITH_ENV(
        "MnnvlRailOn_SameHost_AsicMatch",
        []()
        {
            auto* sys      = makeSystemWithNets({
                {        0,        0,         0xAA,         1, 0xCAFE},
                {0, 1, 0xBB, 1, 0xCAFE}, // same pciId, but different asic on same host
            }
                    );
            auto* g        = makeGraph(NCCL_TOPO_PATTERN_RING, /*crossNic=*/0);
            auto* startNet = &sys->nodes[NET].nodes[0];
            EXPECT_FALSE(ncclTopoSearchCheckNet(sys, g, startNet, 1, 0));
            delete g;
            delete sys;
    },
        {{"NCCL_MNNVL_RAIL_PER_HOST", "1"}});
}

// ----------------------------------------------------------------------------
// PATTERN_BALANCED_TREE: step==0 ignores the back-net check, step!=0 requires
// matching graph->inter[nChannels*2 + 1].
// ----------------------------------------------------------------------------
TEST(SearchCheckNet, BalancedTree_Step0_AcceptsRegardlessOfPriorInter)
{
    RUN_ISOLATED_TEST(
        "BalancedTree_Step0_AcceptsRegardlessOfPriorInter",
        []()
        {
            auto* sys = makeSystemWithNets({
                {0, 0, 0xAA, 1, 0xCC},
                {0, 1, 0xBB, 2, 0xDD},
            });
            auto* g   = makeGraph(NCCL_TOPO_PATTERN_BALANCED_TREE, /*crossNic=*/1, /*nChannels=*/0);
            g->inter[1]    = sys->nodes[NET].nodes[0].id; // not the candidate's id
            auto* startNet = &sys->nodes[NET].nodes[0];
            EXPECT_TRUE(ncclTopoSearchCheckNet(sys, g, startNet, /*n=*/1, /*step=*/0));
            delete g;
            delete sys;
        });
}

TEST(SearchCheckNet, BalancedTree_StepNonZero_MatchingInter_Accepts)
{
    RUN_ISOLATED_TEST(
        "BalancedTree_StepNonZero_MatchingInter_Accepts",
        []()
        {
            auto* sys = makeSystemWithNets({
                {0, 0, 0xAA, 1, 0xCC},
                {0, 1, 0xBB, 2, 0xDD},
            });
            auto* g   = makeGraph(NCCL_TOPO_PATTERN_BALANCED_TREE, /*crossNic=*/1, /*nChannels=*/0);
            g->inter[1]    = sys->nodes[NET].nodes[1].id; // require candidate to be net id #1
            auto* startNet = &sys->nodes[NET].nodes[0];
            EXPECT_TRUE(ncclTopoSearchCheckNet(sys, g, startNet, /*n=*/1, /*step=*/1));
            delete g;
            delete sys;
        });
}

TEST(SearchCheckNet, BalancedTree_StepNonZero_NonMatchingInter_Rejects)
{
    RUN_ISOLATED_TEST(
        "BalancedTree_StepNonZero_NonMatchingInter_Rejects",
        []()
        {
            auto* sys = makeSystemWithNets({
                {0, 0, 0xAA, 1, 0xCC},
                {0, 1, 0xBB, 2, 0xDD},
            });
            auto* g   = makeGraph(NCCL_TOPO_PATTERN_BALANCED_TREE, /*crossNic=*/1, /*nChannels=*/0);
            g->inter[1]    = sys->nodes[NET].nodes[0].id;
            auto* startNet = &sys->nodes[NET].nodes[0];
            EXPECT_FALSE(ncclTopoSearchCheckNet(sys, g, startNet, /*n=*/1, /*step=*/1));
            delete g;
            delete sys;
        });
}

} // namespace RcclUnitTesting
