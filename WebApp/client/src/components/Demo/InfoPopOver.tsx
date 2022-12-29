import {
  Popover,
  PopoverTrigger,
  PopoverContent,
  PopoverBody,
  PopoverArrow,
  PopoverCloseButton,
  UnorderedList,
  ListItem,
} from "@chakra-ui/react";
import CustomButton from "@components/Fixed/CustomButton";

const InfoPopOver = (props) => {
  return (
    <Popover>
      <PopoverTrigger>
        <CustomButton btnText={props.btnText} mt={2} ml={4} />
      </PopoverTrigger>
      <PopoverContent>
        <PopoverArrow />
        <PopoverCloseButton />
        <PopoverBody>
          <UnorderedList>
            <ListItem fontSize="md">Choose 1 among 3 options below.</ListItem>
            <ListItem fontSize="md">
              Once your targeted file is rendered to the screen, either by audio
              or video preview, click &apos;Choose this input&apos; to proceed.
            </ListItem>
            <ListItem fontSize="md">
              After that, if you wish to change the input file, click
              &apos;Different input&apos; in the next page to return to this tab
              menu.
            </ListItem>
          </UnorderedList>
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
};

export default InfoPopOver;
