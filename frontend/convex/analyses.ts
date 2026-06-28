import { mutation, query } from "./_generated/server";
import { v } from "convex/values";

async function findJob(ctx: any, externalJobId: string) {
  return await ctx.db
    .query("analysisJobs")
    .withIndex("by_externalJobId", (q: any) => q.eq("externalJobId", externalJobId))
    .unique();
}

function compactObject<T extends Record<string, unknown>>(value: T) {
  return Object.fromEntries(
    Object.entries(value).filter(([, item]) => item !== undefined),
  ) as Partial<T>;
}

export const createJob = mutation({
  args: {
    externalJobId: v.string(),
    sourceType: v.string(),
    targetCustomer: v.string(),
    demographicTarget: v.optional(v.string()),
    goal: v.string(),
  },
  handler: async (ctx, args) => {
    const now = Date.now();
    const existing = await findJob(ctx, args.externalJobId);
    const jobFields: {
      externalJobId: string;
      sourceType: string;
      targetCustomer: string;
      demographicTarget?: string;
      goal: string;
    } = {
      externalJobId: args.externalJobId,
      sourceType: args.sourceType,
      targetCustomer: args.targetCustomer,
      goal: args.goal,
    };
    if (args.demographicTarget !== undefined) {
      jobFields.demographicTarget = args.demographicTarget;
    }

    if (existing) {
      await ctx.db.patch(existing._id, {
        ...jobFields,
        status: "running",
        updatedAt: now,
      });
      return existing._id;
    }

    return await ctx.db.insert("analysisJobs", {
      ...jobFields,
      status: "running",
      createdAt: now,
      updatedAt: now,
    });
  },
});

export const addEvent = mutation({
  args: {
    externalJobId: v.string(),
    event: v.string(),
    agent: v.optional(v.string()),
    label: v.optional(v.string()),
    payload: v.any(),
  },
  handler: async (ctx, args) => {
    const now = Date.now();
    const eventFields: {
      externalJobId: string;
      event: string;
      agent?: string;
      label?: string;
      payload: any;
      createdAt: number;
    } = {
      externalJobId: args.externalJobId,
      event: args.event,
      payload: args.payload,
      createdAt: now,
    };
    if (args.agent !== undefined) eventFields.agent = args.agent;
    if (args.label !== undefined) eventFields.label = args.label;
    await ctx.db.insert("analysisEvents", eventFields);

    const job = await findJob(ctx, args.externalJobId);
    if (!job) return;

    const patch: Record<string, unknown> = { updatedAt: now };
    if (args.event === "capture_done" && typeof args.payload.image_url === "string") {
      patch.originalImageUrl = args.payload.image_url;
    }
    if (args.event === "heatmap_ready" && typeof args.payload.heatmap_url === "string") {
      patch.heatmapUrl = args.payload.heatmap_url;
    }
    if (args.event === "scored" && typeof args.payload.fixate_score === "number") {
      patch.baselineScore = args.payload.fixate_score;
    }
    if (args.event === "job_complete" && typeof args.payload.final_score === "number") {
      patch.finalScore = args.payload.final_score;
      patch.status = "complete";
    }

    await ctx.db.patch(job._id, patch);
  },
});

export const completeJob = mutation({
  args: {
    externalJobId: v.string(),
    baselineScore: v.optional(v.number()),
    finalScore: v.optional(v.number()),
    heatmapUrl: v.optional(v.string()),
    originalImageUrl: v.optional(v.string()),
    bestImageUrl: v.optional(v.string()),
    selectedAudience: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    const job = await findJob(ctx, args.externalJobId);
    if (!job) return null;

    await ctx.db.patch(job._id, compactObject({
      status: "complete",
      baselineScore: args.baselineScore,
      finalScore: args.finalScore,
      heatmapUrl: args.heatmapUrl,
      originalImageUrl: args.originalImageUrl,
      bestImageUrl: args.bestImageUrl,
      selectedAudience: args.selectedAudience,
      updatedAt: Date.now(),
    }));
    return job._id;
  },
});

export const failJob = mutation({
  args: {
    externalJobId: v.string(),
    errorMessage: v.string(),
  },
  handler: async (ctx, args) => {
    const job = await findJob(ctx, args.externalJobId);
    if (!job) return null;

    await ctx.db.patch(job._id, {
      status: "failed",
      errorMessage: args.errorMessage,
      updatedAt: Date.now(),
    });
    return job._id;
  },
});

export const listJobs = query({
  args: {},
  handler: async (ctx) => {
    return await ctx.db.query("analysisJobs").order("desc").take(20);
  },
});

export const listEvents = query({
  args: {
    externalJobId: v.string(),
  },
  handler: async (ctx, args) => {
    return await ctx.db
      .query("analysisEvents")
      .withIndex("by_externalJobId", (q) => q.eq("externalJobId", args.externalJobId))
      .collect();
  },
});
