import { defineSchema, defineTable } from "convex/server";
import { v } from "convex/values";

export default defineSchema({
  analysisJobs: defineTable({
    externalJobId: v.string(),
    sourceType: v.string(),
    targetCustomer: v.string(),
    demographicTarget: v.optional(v.string()),
    goal: v.string(),
    status: v.union(v.literal("running"), v.literal("complete"), v.literal("failed")),
    baselineScore: v.optional(v.number()),
    finalScore: v.optional(v.number()),
    heatmapUrl: v.optional(v.string()),
    originalImageUrl: v.optional(v.string()),
    bestImageUrl: v.optional(v.string()),
    selectedAudience: v.optional(v.string()),
    errorMessage: v.optional(v.string()),
    createdAt: v.number(),
    updatedAt: v.number(),
  }).index("by_externalJobId", ["externalJobId"]),

  analysisEvents: defineTable({
    externalJobId: v.string(),
    event: v.string(),
    agent: v.optional(v.string()),
    label: v.optional(v.string()),
    payload: v.any(),
    createdAt: v.number(),
  }).index("by_externalJobId", ["externalJobId"]),
});
