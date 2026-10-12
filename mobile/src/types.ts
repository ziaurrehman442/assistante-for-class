export type RecordingStatus = "PENDING_SYNC" | "SYNCING" | "SYNCED";

export type StudyMaterials = {
  detailed_class_notes: {
    main_topics: string[];
    definitions: string[];
    teacher_explanations: string[];
    qa: { question: string; answer: string }[];
  };
  easy_notes: string[];
  revision_quiz: {
    mcqs: { question: string; options: string[]; answer: string }[];
    conceptual_questions: string[];
  };
};

export type SpeakerTurn = {
  speaker_id: string;
  speaker_name: string;
  overlapping_speakers?: string[];
  start: number;
  end: number;
  text: string;
};

export type RecordingSpeaker = {
  id: string;
  display_name: string;
  is_named: boolean;
};

export type SubjectFolder = {
  id: string;
  name: string;
  createdAt: string;
};

export type ClassRecording = {
  id: string;
  title: string;
  uri: string;
  createdAt: string;
  durationSeconds: number;
  status: RecordingStatus;
  subjectId?: string;
  processingJobId?: string;
  attempt?: number;
  transcript?: string;
  dialogue?: SpeakerTurn[];
  speakers?: RecordingSpeaker[];
  materials?: StudyMaterials;
  lastError?: string;
};
