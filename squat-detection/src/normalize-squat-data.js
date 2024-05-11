export function normalizeSquatData(trainingInputs) {
  const normalizedInputs = trainingInputs.reduce((oldData, poses) => {
    const newPoses = poses.reduce((oldData, pose) => {
      oldData.push(pose.x / 640);
      oldData.push(pose.y / 480);
      return oldData;
    }, []);
    oldData.push(newPoses);
    return oldData;
  }, []);

  return normalizedInputs;
}
