import VideoItem from "./VideoItem";
import { SimpleGrid } from "@chakra-ui/react";

const VideoList = ({ videos, handleVideoSelect }) => {
  const renderedVideos = videos.map((video) => {
    return (
      <VideoItem
        key={video.id.videoId}
        video={video}
        handleVideoSelect={handleVideoSelect}
      />
    );
  });

  return (
    <div>
      <SimpleGrid columns={{ base: 1, md: 3 }} spacing={{ base: 5, lg: 8 }}>
        {renderedVideos}
      </SimpleGrid>
    </div>
  );
};
export default VideoList;
