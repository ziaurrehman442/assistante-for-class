import AsyncStorage from "@react-native-async-storage/async-storage";
import * as FileSystem from "expo-file-system/legacy";

import { ClassRecording, SubjectFolder } from "../types";

const RECORDINGS_KEY = "@class-notes/recordings-v1";
const SUBJECTS_KEY = "@class-notes/subjects-v1";
const RECORDINGS_DIRECTORY = `${FileSystem.documentDirectory}class-recordings/`;

export async function prepareRecordingDirectory(): Promise<void> {
  if (!FileSystem.documentDirectory) {
    throw new Error("Device document storage is unavailable.");
  }
  const directory = await FileSystem.getInfoAsync(RECORDINGS_DIRECTORY);
  if (!directory.exists) {
    await FileSystem.makeDirectoryAsync(RECORDINGS_DIRECTORY, {
      intermediates: true,
    });
  }
}

export function createRecordingPath(id: string): string {
  return `${RECORDINGS_DIRECTORY}${id}.m4a`;
}

export async function moveRecordingToLocalStorage(
  sourceUri: string,
  destinationUri: string,
): Promise<void> {
  await prepareRecordingDirectory();
  await FileSystem.moveAsync({ from: sourceUri, to: destinationUri });
}

export async function loadRecordings(): Promise<ClassRecording[]> {
  const stored = await AsyncStorage.getItem(RECORDINGS_KEY);
  if (!stored) {
    return [];
  }

  const recordings = JSON.parse(stored) as ClassRecording[];
  let changed = false;
  const restored = recordings.map((recording) => {
    if (recording.status !== "SYNCING") {
      return recording;
    }
    changed = true;
    return { ...recording, status: "PENDING_SYNC" as const };
  });

  if (changed) {
    await saveRecordings(restored);
  }
  return restored;
}

export async function loadSubjectFolders(): Promise<SubjectFolder[]> {
  const stored = await AsyncStorage.getItem(SUBJECTS_KEY);
  const subjects: SubjectFolder[] = stored ? JSON.parse(stored) as SubjectFolder[] : [];
  const recordings = await loadRecordings();
  const legacyRecordings = recordings.filter((recording) => !recording.subjectId);
  if (legacyRecordings.length === 0) {
    return subjects;
  }

  const defaultSubject: SubjectFolder = subjects.find((subject) => subject.id === "general") ?? {
    id: "general",
    name: "General",
    createdAt: new Date().toISOString(),
  };
  const updatedSubjects = subjects.some((subject) => subject.id === defaultSubject.id)
    ? subjects
    : [defaultSubject, ...subjects];
  await AsyncStorage.multiSet([
    [SUBJECTS_KEY, JSON.stringify(updatedSubjects)],
    [
      RECORDINGS_KEY,
      JSON.stringify(
        recordings.map((recording) =>
          recording.subjectId ? recording : { ...recording, subjectId: defaultSubject.id },
        ),
      ),
    ],
  ]);
  return updatedSubjects;
}

export async function createSubjectFolder(name: string): Promise<SubjectFolder> {
  const normalizedName = name.trim();
  if (!normalizedName || normalizedName.length > 50) {
    throw new Error("Subject name must contain 1 to 50 characters.");
  }

  const subjects = await loadSubjectFolders();
  if (subjects.some((subject) => subject.name.toLocaleLowerCase() === normalizedName.toLocaleLowerCase())) {
    throw new Error("A folder with that subject name already exists.");
  }

  const subject: SubjectFolder = {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    name: normalizedName,
    createdAt: new Date().toISOString(),
  };
  await AsyncStorage.setItem(SUBJECTS_KEY, JSON.stringify([...subjects, subject]));
  return subject;
}

export async function saveRecordings(
  recordings: ClassRecording[],
): Promise<void> {
  await AsyncStorage.setItem(RECORDINGS_KEY, JSON.stringify(recordings));
}
